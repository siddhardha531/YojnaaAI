import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import dotenv from 'dotenv';
import express, { type NextFunction, type Request, type Response } from 'express';
import multer from 'multer';
import { extractText } from 'unpdf';
import { INITIAL_SCHEMES, SCHEME_CATEGORIES } from './src/data/schemesData.ts';
import type {
  AdminDashboardStats,
  AdminUser,
  AuthUser,
  ChatMessage,
  CitizenDocument,
  ConsolidatedPublicScheme,
  EligibilityHistoryEntry,
  EligibilityProfile,
  ExtractedSchemeDraft,
  PdfExtractionJob,
  PublicPdfAnalysisResult,
  PublicPdfJobStatus,
  Scheme,
  SchemeAttachedDocument,
  UserRole,
} from './src/types/scheme.ts';
import {
  extractSchemeFromPdf,
  generateGroundedAssistantReply,
  getGeminiClient,
} from './server/geminiService.ts';
import { consolidateAdminGroup } from './server/multiDocConsolidator.ts';
import {
  addDocumentsToPublicSession,
  chatAboutPublicPdf,
  createPublicAnalysisSession,
  getPublicAnalysisSession,
  getPublicPdfAnalysisJob,
  PdfAnalysisError,
  removePublicSessionDocument,
  retryPublicSessionDocument,
  sanitizeErrorText,
  startPublicPdfAnalysisJob,
} from './server/publicPdfService.ts';
import { smartSearchSchemes, syncRagIndex } from './server/ragEngine.ts';
import {
  appendNotification,
  computeSha256,
  ensureStorageDirs,
  findDuplicateScheme,
  getPersonalUploadDir,
  getUploadDir,
  hashPassword,
  readNotifications,
  readPdfJobs,
  readPersonalFiles,
  readSavedIds,
  readSchemes,
  readUserWorkspace,
  readUsers,
  slugifySchemeId,
  verifyPassword,
  writeDocuments,
  writePdfJobs,
  writePersonalFiles,
  writeSavedIds,
  writeSchemes,
  writeUserWorkspace,
  writeUsers,
} from './server/store.ts';

dotenv.config();
ensureStorageDirs();

const PORT = Number(process.env.PORT) || 3000;
const ADMIN_EMAIL = (process.env.ADMIN_EMAIL || 'admin@yojana.gov.in').toLowerCase();
const SESSION_SECRET =
  process.env.SESSION_SECRET || 'yojanaai-local-express-dev-secret';

const app = express();
app.use('/api', (req: Request, res: Response, next: NextFunction) => {
  const origin = req.headers.origin;
  if (origin) {
    res.setHeader('Access-Control-Allow-Origin', origin);
    res.setHeader('Vary', 'Origin');
  } else {
    res.setHeader('Access-Control-Allow-Origin', '*');
  }
  res.setHeader('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  if (req.method === 'OPTIONS') {
    res.status(204).end();
    return;
  }
  next();
});
app.use(express.json({ limit: '4mb' }));

// Signed token helpers for User & Admin RBAC Auth
function signAuthToken(user: AuthUser): string {
  const payload = Buffer.from(
    JSON.stringify({
      id: user.id,
      email: user.email,
      name: user.name,
      role: user.role,
      exp: Date.now() + 1000 * 60 * 60 * 12,
    })
  ).toString('base64url');
  const sig = crypto
    .createHmac('sha256', SESSION_SECRET)
    .update(payload)
    .digest('base64url');
  return `${payload}.${sig}`;
}

function verifyAuthToken(token: string): AuthUser | null {
  const parts = token.split('.');
  if (parts.length !== 2) return null;
  const [payload, sig] = parts;
  const expectedSig = crypto
    .createHmac('sha256', SESSION_SECRET)
    .update(payload)
    .digest('base64url');

  const sigBuf = Buffer.from(sig);
  const expectedBuf = Buffer.from(expectedSig);
  if (
    sigBuf.length !== expectedBuf.length ||
    !crypto.timingSafeEqual(sigBuf, expectedBuf)
  ) {
    return null;
  }

  try {
    const decoded = JSON.parse(
      Buffer.from(payload, 'base64url').toString('utf-8')
    ) as { id?: string; email?: string; name?: string; role?: UserRole; exp?: number };
    if (!decoded.email || !decoded.exp || decoded.exp < Date.now()) {
      return null;
    }
    const role: UserRole =
      decoded.role === 'admin' || decoded.email.toLowerCase() === ADMIN_EMAIL
        ? 'admin'
        : 'user';
    return {
      id: decoded.id || (role === 'admin' ? 'admin-root' : decoded.email),
      email: decoded.email,
      name: decoded.name || (role === 'admin' ? 'Portal Administrator' : 'Citizen User'),
      role,
    };
  } catch {
    return null;
  }
}

interface AuthedRequest extends Request {
  authUser?: AuthUser;
  adminUser?: AdminUser;
}

function extractBearerUser(req: Request): AuthUser | null {
  const authHeader = req.headers.authorization || '';
  const token = authHeader.startsWith('Bearer ') ? authHeader.slice(7).trim() : '';
  return token ? verifyAuthToken(token) : null;
}

function resolveWorkspaceOwner(req: Request): string {
  const user = extractBearerUser(req);
  return user ? user.id : 'guest';
}

function requireAuth(req: AuthedRequest, res: Response, next: NextFunction): void {
  const user = extractBearerUser(req);
  if (!user) {
    res.status(401).json({ error: 'Authentication required. Please sign in.' });
    return;
  }
  req.authUser = user;
  next();
}

function requireAdmin(req: AuthedRequest, res: Response, next: NextFunction): void {
  const user = extractBearerUser(req);
  if (!user) {
    res.status(401).json({ error: 'Administrator authentication required.' });
    return;
  }
  if (user.role !== 'admin') {
    res.status(403).json({
      error: 'Forbidden: Scheme modifications are restricted to authorized administrators.',
    });
    return;
  }
  req.authUser = user;
  req.adminUser = { id: user.id, email: user.email, name: user.name, role: 'admin' };
  next();
}

function hasValidPdfSignature(buffer: Buffer): boolean {
  return buffer.length >= 5 && buffer.subarray(0, 1024).indexOf('%PDF-') !== -1;
}

function hasValidPersonalDocSignature(buffer: Buffer, ext: string): boolean {
  if (ext === '.pdf') {
    return hasValidPdfSignature(buffer);
  }
  if (ext === '.png') {
    return (
      buffer.length >= 4 &&
      buffer[0] === 0x89 &&
      buffer[1] === 0x50 &&
      buffer[2] === 0x4e &&
      buffer[3] === 0x47
    );
  }
  if (ext === '.jpg' || ext === '.jpeg') {
    return (
      buffer.length >= 3 &&
      buffer[0] === 0xff &&
      buffer[1] === 0xd8 &&
      buffer[2] === 0xff
    );
  }
  return false;
}

// Multer configuration for PDF upload storage
const upload = multer({
  storage: multer.diskStorage({
    destination: (_req, _file, cb) => {
      cb(null, getUploadDir());
    },
    filename: (_req, file, cb) => {
      const safeName = file.originalname
        .replace(/[^a-zA-Z0-9._-]/g, '_')
        .replace(/_{2,}/g, '_');
      cb(null, `${Date.now()}_${safeName}`);
    },
  }),
  limits: {
    fileSize: 15 * 1024 * 1024,
  },
  fileFilter: (_req, file, cb) => {
    const isPdf =
      file.mimetype === 'application/pdf' ||
      file.originalname.toLowerCase().endsWith('.pdf');
    if (!isPdf) {
      cb(new Error('Only PDF files (.pdf) are supported.'));
      return;
    }
    cb(null, true);
  },
});

// Multer configuration for personal citizen documents (PDF, PNG, JPG up to 5MB)
const personalUpload = multer({
  storage: multer.diskStorage({
    destination: (_req, _file, cb) => {
      cb(null, getPersonalUploadDir());
    },
    filename: (_req, file, cb) => {
      const safeName = file.originalname
        .replace(/[^a-zA-Z0-9._-]/g, '_')
        .replace(/_{2,}/g, '_');
      cb(null, `${Date.now()}_${safeName}`);
    },
  }),
  limits: {
    fileSize: 5 * 1024 * 1024,
  },
  fileFilter: (_req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    const allowedExt = ['.pdf', '.png', '.jpg', '.jpeg'].includes(ext);
    const allowedMime = [
      'application/pdf',
      'image/png',
      'image/jpeg',
      'image/jpg',
    ].includes(file.mimetype);
    if (!allowedExt || !allowedMime) {
      cb(new Error('Only PDF, PNG, and JPG files up to 5MB are allowed.'));
      return;
    }
    cb(null, true);
  },
});

// Ephemeral in-memory Multer for Public PDF Upload & Analysis (never stored on disk)
const publicPdfUpload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 10 * 1024 * 1024,
  },
  fileFilter: (_req, file, cb) => {
    const mime = (file.mimetype || '').toLowerCase();
    const isPdf =
      mime === 'application/pdf' ||
      mime === 'application/x-pdf' ||
      file.originalname.toLowerCase().endsWith('.pdf');
    if (!isPdf) {
      const err = Object.assign(
        new Error('Only PDF files (.pdf) up to 10 MB are supported.'),
        { statusCode: 415, code: 'UNSUPPORTED_FILE' }
      );
      cb(err);
      return;
    }
    cb(null, true);
  },
});

// Serve stored scheme PDFs safely
app.get('/api/pdfs/:filename', (req: Request, res: Response) => {
  const safeFilename = path.basename(req.params.filename);
  const fullPath = path.join(getUploadDir(), safeFilename);
  if (!fs.existsSync(fullPath)) {
    res.status(404).json({ error: 'PDF document not found.' });
    return;
  }
  res.setHeader('Content-Type', 'application/pdf');
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('Content-Disposition', `inline; filename="${safeFilename}"`);
  fs.createReadStream(fullPath).pipe(res);
});

// Health
app.get('/api/health', (_req: Request, res: Response) => {
  res.json({
    ok: true,
    schemesCount: readSchemes().length,
    geminiConfigured: Boolean(getGeminiClient()),
  });
});

// Unified Citizen & Admin Authentication
app.post('/api/auth/register', (req: Request, res: Response) => {
  const { name, email, password } = req.body as {
    name?: string;
    email?: string;
    password?: string;
  };
  const cleanName = (name || '').trim();
  const normalizedEmail = (email || '').trim().toLowerCase();
  const cleanPassword = password || '';

  if (!cleanName || cleanName.length < 2) {
    res.status(400).json({ error: 'Full name (at least 2 characters) is required.' });
    return;
  }
  if (!normalizedEmail || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(normalizedEmail)) {
    res.status(400).json({ error: 'Please enter a valid email address.' });
    return;
  }
  if (cleanPassword.length < 6) {
    res.status(400).json({ error: 'Password must be at least 6 characters.' });
    return;
  }

  const users = readUsers();
  if (users.some((u) => u.email.toLowerCase() === normalizedEmail)) {
    res.status(409).json({ error: 'An account with this email already exists.' });
    return;
  }

  const createdAt = new Date().toISOString();
  const newUserRecord = {
    id: `user-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    email: normalizedEmail,
    name: cleanName,
    role: 'user' as const,
    passwordHash: hashPassword(cleanPassword),
    createdAt,
  };
  writeUsers([...users, newUserRecord]);

  const publicUser: AuthUser = {
    id: newUserRecord.id,
    email: newUserRecord.email,
    name: newUserRecord.name,
    role: 'user',
    createdAt,
  };
  const token = signAuthToken(publicUser);
  res.status(201).json({ token, user: publicUser });
});

app.post('/api/auth/login', (req: Request, res: Response) => {
  const { email, password } = req.body as { email?: string; password?: string };
  const normalizedEmail = (email || '').trim().toLowerCase();
  const cleanPassword = password || '';

  if (!normalizedEmail || !cleanPassword) {
    res.status(400).json({ error: 'Email and password are required.' });
    return;
  }

  const users = readUsers();
  const found = users.find((u) => u.email.toLowerCase() === normalizedEmail);
  if (!found || !verifyPassword(cleanPassword, found.passwordHash)) {
    res.status(401).json({ error: 'Invalid email or password.' });
    return;
  }

  const publicUser: AuthUser = {
    id: found.id,
    email: found.email,
    name: found.name,
    role: found.role,
    createdAt: found.createdAt,
  };
  const token = signAuthToken(publicUser);
  res.json({ token, user: publicUser });
});

app.get('/api/auth/session', requireAuth, (req: AuthedRequest, res: Response) => {
  res.json({ user: req.authUser });
});

// Public Scheme Catalog & Intelligent Natural-Language Search
function filterPublicSchemes(all: Scheme[]): Scheme[] {
  return all.filter(
    (s) =>
      s.publicationStatus !== 'Draft' &&
      s.verificationStatus !== 'Pending Review' &&
      s.verificationStatus !== 'Rejected'
  );
}

app.get('/api/schemes', (_req: Request, res: Response) => {
  res.json({ schemes: filterPublicSchemes(readSchemes()) });
});

app.post('/api/schemes/search', async (req: Request, res: Response) => {
  const { query } = req.body as { query?: string };
  if (!query || typeof query !== 'string' || !query.trim()) {
    res.status(400).json({ error: 'Search query is required.' });
    return;
  }
  const result = await smartSearchSchemes(
    query.trim(),
    filterPublicSchemes(readSchemes()),
    getGeminiClient()
  );
  res.json(result);
});

// Public Ephemeral PDF Analysis & Q&A (completely separated from official catalog)
app.post(
  '/api/public/pdf/analyze',
  publicPdfUpload.single('file'),
  (req: Request, res: Response) => {
    try {
      if (!req.file) {
        res.status(400).json({
          error: 'Please select a PDF file to analyze.',
          code: 'NO_FILE',
        });
        return;
      }
      if (req.file.size === 0 || req.file.buffer.length === 0) {
        res.status(422).json({
          error: 'The uploaded PDF file is empty (0 bytes). Please upload a valid scheme PDF.',
          code: 'INVALID_OR_CORRUPTED_PDF',
        });
        return;
      }
      if (!hasValidPdfSignature(req.file.buffer)) {
        res.status(422).json({
          error:
            'The uploaded file is not a valid PDF document (%PDF- header missing). Please check the file and try again.',
          code: 'INVALID_OR_CORRUPTED_PDF',
        });
        return;
      }
      const job = startPublicPdfAnalysisJob(
        req.file.buffer,
        req.file.originalname,
        req.file.size
      );
      res.setHeader('Cache-Control', 'no-store');
      res.status(202).json({ job });
    } catch (error) {
      if (error instanceof PdfAnalysisError) {
        console.error(
          `[public-pdf/analyze] ${error.code} (${error.status}):`,
          sanitizeErrorText(error.message)
        );
        res.status(error.status).json({
          error: error.message,
          code: error.code,
        });
        return;
      }
      const rawMsg = error instanceof Error ? error.message : String(error);
      const safeMsg = sanitizeErrorText(rawMsg);
      console.error('[public-pdf/analyze] unexpected error:', safeMsg);
      res.status(500).json({
        error: safeMsg || 'Unable to start PDF analysis job due to a server error.',
        code: 'SERVER_ERROR',
      });
    }
  }
);

app.get('/api/public/pdf/jobs/:jobId', (req: Request, res: Response) => {
  res.setHeader('Cache-Control', 'no-store');
  const jobId = (req.params.jobId || '').trim();
  if (!jobId) {
    res.status(400).json({ error: 'Job ID is required.', code: 'INVALID_REQUEST' });
    return;
  }
  const job = getPublicPdfAnalysisJob(jobId);
  if (!job) {
    res.status(404).json({
      error: 'Analysis job not found or expired. Please upload the PDF again.',
      code: 'JOB_NOT_FOUND',
    });
    return;
  }
  res.json({ job });
});

// Public Multi-Document Sessions (ephemeral in-memory only)
app.post('/api/public/pdf/sessions', (_req: Request, res: Response) => {
  res.setHeader('Cache-Control', 'no-store');
  const session = createPublicAnalysisSession();
  res.status(201).json({ session });
});

app.get('/api/public/pdf/sessions/:sessionId', (req: Request, res: Response) => {
  res.setHeader('Cache-Control', 'no-store');
  const session = getPublicAnalysisSession((req.params.sessionId || '').trim());
  if (!session) {
    res.status(404).json({
      error: 'Temporary analysis session not found or expired. Please upload the PDF files again.',
      code: 'SESSION_NOT_FOUND',
    });
    return;
  }
  res.json({ session });
});

app.post(
  '/api/public/pdf/sessions/:sessionId/documents',
  publicPdfUpload.fields([
    { name: 'files', maxCount: 10 },
    { name: 'file', maxCount: 1 },
  ]),
  (req: Request, res: Response) => {
    res.setHeader('Cache-Control', 'no-store');
    const sessionId = (req.params.sessionId || '').trim();
    const existingSession = getPublicAnalysisSession(sessionId);
    if (!existingSession) {
      res.status(404).json({
        error: 'Temporary analysis session not found or expired.',
        code: 'SESSION_NOT_FOUND',
      });
      return;
    }

    const fileGroups = (req.files || {}) as Record<string, Express.Multer.File[]>;
    const uploadedFiles = [
      ...(fileGroups.files || []),
      ...(fileGroups.file || []),
    ];

    if (uploadedFiles.length === 0) {
      res.status(400).json({
        error: 'Please select at least one PDF file to upload.',
        code: 'NO_FILE',
      });
      return;
    }

    const validFiles: Array<{ buffer: Buffer; fileName: string; fileSize: number }> = [];
    const rejectedFiles: Array<{ fileName: string; code: string; message: string }> = [];

    for (const f of uploadedFiles) {
      if (f.size === 0 || f.buffer.length === 0) {
        rejectedFiles.push({
          fileName: f.originalname,
          code: 'INVALID_OR_CORRUPTED_PDF',
          message: `"${f.originalname}" is empty (0 bytes).`,
        });
        continue;
      }
      if (!hasValidPdfSignature(f.buffer)) {
        rejectedFiles.push({
          fileName: f.originalname,
          code: 'INVALID_OR_CORRUPTED_PDF',
          message: `"${f.originalname}" is not a valid PDF (%PDF- header missing).`,
        });
        continue;
      }
      validFiles.push({
        buffer: f.buffer,
        fileName: f.originalname,
        fileSize: f.size,
      });
    }

    const result = addDocumentsToPublicSession(sessionId, validFiles);
    if (!result) {
      res.status(404).json({ error: 'Session expired.', code: 'SESSION_NOT_FOUND' });
      return;
    }

    res.status(202).json({
      session: result.session,
      acceptedJobs: result.acceptedJobs,
      duplicates: result.duplicates,
      rejectedFiles,
    });
  }
);

app.post(
  '/api/public/pdf/sessions/:sessionId/documents/:documentId/retry',
  publicPdfUpload.single('file'),
  (req: Request, res: Response) => {
    res.setHeader('Cache-Control', 'no-store');
    const sessionId = (req.params.sessionId || '').trim();
    const documentId = (req.params.documentId || '').trim();
    try {
      let replacementBuffer: Buffer | undefined;
      if (req.file) {
        if (req.file.size === 0 || !hasValidPdfSignature(req.file.buffer)) {
          res.status(422).json({
            error: 'Replacement file is not a valid PDF document.',
            code: 'INVALID_OR_CORRUPTED_PDF',
          });
          return;
        }
        replacementBuffer = req.file.buffer;
      }

      const result = retryPublicSessionDocument(sessionId, documentId, replacementBuffer);
      if (!result) {
        res.status(404).json({
          error: 'Document or session not found.',
          code: 'SESSION_NOT_FOUND',
        });
        return;
      }
      res.status(202).json(result);
    } catch (error) {
      if (error instanceof PdfAnalysisError) {
        res.status(error.status).json({ error: error.message, code: error.code });
        return;
      }
      const msg = sanitizeErrorText(error instanceof Error ? error.message : String(error));
      res.status(500).json({ error: msg || 'Retry failed.', code: 'SERVER_ERROR' });
    }
  }
);

app.delete(
  '/api/public/pdf/sessions/:sessionId/documents/:documentId',
  (req: Request, res: Response) => {
    res.setHeader('Cache-Control', 'no-store');
    const session = removePublicSessionDocument(
      (req.params.sessionId || '').trim(),
      (req.params.documentId || '').trim()
    );
    if (!session) {
      res.status(404).json({ error: 'Session not found.', code: 'SESSION_NOT_FOUND' });
      return;
    }
    res.json({ session });
  }
);

app.post('/api/public/pdf/chat', async (req: Request, res: Response) => {
  const { question, history, sessionId, analysis, analyses, consolidated, profile } = req.body as {
    question?: string;
    history?: ChatMessage[];
    sessionId?: string;
    analysis?: Pick<PublicPdfAnalysisResult, 'fileName' | 'pages' | 'extracted'>;
    analyses?: Array<Pick<PublicPdfAnalysisResult, 'fileName' | 'pages' | 'extracted'>>;
    consolidated?: ConsolidatedPublicScheme | null;
    profile?: EligibilityProfile;
  };
  if (!question || !question.trim()) {
    res.status(400).json({ error: 'Question is required.', code: 'INVALID_REQUEST' });
    return;
  }
  if (!sessionId && !analysis && (!Array.isArray(analyses) || analyses.length === 0)) {
    res.status(400).json({
      error: 'Analyzed PDF context or active sessionId is required.',
      code: 'INVALID_REQUEST',
    });
    return;
  }
  try {
    const reply = await chatAboutPublicPdf(
      question.trim(),
      Array.isArray(history) ? history : [],
      { sessionId, analysis, analyses, consolidated },
      profile
    );
    res.json(reply);
  } catch (error) {
    if (error instanceof PdfAnalysisError) {
      console.error(
        `[public-pdf/chat] ${error.code} (${error.status}):`,
        sanitizeErrorText(error.message)
      );
      res.status(error.status).json({
        error: error.message,
        code: error.code,
      });
      return;
    }
    const rawMsg = error instanceof Error ? error.message : String(error);
    const safeMsg = sanitizeErrorText(rawMsg);
    console.error('[public-pdf/chat] unexpected error:', safeMsg);
    res.status(500).json({
      error: safeMsg || 'Unable to answer questions about the PDF right now.',
      code: 'SERVER_ERROR',
    });
  }
});

// Notifications Feed
app.get('/api/notifications', (req: Request, res: Response) => {
  const ownerId = resolveWorkspaceOwner(req);
  const workspace = readUserWorkspace(ownerId);
  const notifications = readNotifications();
  const readSet = new Set(workspace.readNotificationIds);
  res.json({
    notifications,
    readIds: workspace.readNotificationIds,
    unreadCount: notifications.filter((n) => !readSet.has(n.id)).length,
  });
});

app.post('/api/notifications/read', (req: Request, res: Response) => {
  const ownerId = resolveWorkspaceOwner(req);
  const { notificationIds } = req.body as { notificationIds?: string[] | 'all' };
  const allNotifications = readNotifications();
  const nextWorkspace = writeUserWorkspace(ownerId, (current) => {
    const nextRead = new Set(current.readNotificationIds);
    if (notificationIds === 'all') {
      allNotifications.forEach((n) => nextRead.add(n.id));
    } else if (Array.isArray(notificationIds)) {
      notificationIds.forEach((id) => nextRead.add(String(id)));
    }
    return {
      ...current,
      readNotificationIds: Array.from(nextRead),
    };
  });
  res.json({ readIds: nextWorkspace.readNotificationIds });
});

// User Dashboard Workspace Endpoints (scoped cleanly by authenticated userId or guest)
app.get('/api/workspace', (req: Request, res: Response) => {
  const ownerId = resolveWorkspaceOwner(req);
  const workspace = readUserWorkspace(ownerId);
  res.json({ workspace, ownerId });
});

app.put('/api/workspace/profile', (req: Request, res: Response) => {
  const ownerId = resolveWorkspaceOwner(req);
  const { profile } = req.body as { profile?: EligibilityProfile };
  if (!profile) {
    res.status(400).json({ error: 'Invalid profile payload.' });
    return;
  }
  const next = writeUserWorkspace(ownerId, (curr) => ({ ...curr, profile }));
  res.json({ profile: next.profile });
});

app.post('/api/workspace/viewed', (req: Request, res: Response) => {
  const ownerId = resolveWorkspaceOwner(req);
  const { schemeId } = req.body as { schemeId?: string };
  if (!schemeId) {
    res.status(400).json({ error: 'schemeId is required.' });
    return;
  }
  const next = writeUserWorkspace(ownerId, (curr) => {
    const filtered = curr.recentlyViewedIds.filter((id) => id !== schemeId);
    return {
      ...curr,
      recentlyViewedIds: [schemeId, ...filtered].slice(0, 12),
    };
  });
  res.json({ recentlyViewedIds: next.recentlyViewedIds });
});

app.post('/api/workspace/eligibility-history', (req: Request, res: Response) => {
  const ownerId = resolveWorkspaceOwner(req);
  const { entry } = req.body as { entry?: EligibilityHistoryEntry };
  if (!entry || !entry.profile) {
    res.status(400).json({ error: 'Invalid eligibility history entry.' });
    return;
  }
  const normalized: EligibilityHistoryEntry = {
    ...entry,
    id: entry.id || `elig-${Date.now()}`,
    evaluatedAt: entry.evaluatedAt || new Date().toISOString(),
  };
  const next = writeUserWorkspace(ownerId, (curr) => ({
    ...curr,
    eligibilityHistory: [normalized, ...curr.eligibilityHistory].slice(0, 20),
  }));
  res.status(201).json({ eligibilityHistory: next.eligibilityHistory });
});

app.put('/api/workspace/chat', (req: Request, res: Response) => {
  const ownerId = resolveWorkspaceOwner(req);
  const { messages } = req.body as { messages?: ChatMessage[] };
  if (!Array.isArray(messages)) {
    res.status(400).json({ error: 'Invalid messages array.' });
    return;
  }
  const next = writeUserWorkspace(ownerId, (curr) => ({
    ...curr,
    chatHistory: messages.slice(-40),
  }));
  res.json({ chatHistory: next.chatHistory });
});

// Protected Personal Document File Uploads (strictly owner-scoped)
app.post(
  '/api/user/documents/upload',
  requireAuth,
  personalUpload.single('file'),
  (req: AuthedRequest, res: Response) => {
    try {
      if (!req.file || !req.authUser) {
        res.status(400).json({ error: 'Please select a document file to upload.' });
        return;
      }

      const ext = path.extname(req.file.originalname).toLowerCase();
      const buffer = fs.readFileSync(req.file.path);
      if (!hasValidPersonalDocSignature(buffer, ext)) {
        fs.unlinkSync(req.file.path);
        res.status(400).json({
          error: 'File content failed security signature validation.',
        });
        return;
      }

      const documentLabel =
        typeof req.body?.documentLabel === 'string' && req.body.documentLabel.trim()
          ? req.body.documentLabel.trim()
          : req.file.originalname;

      const record = {
        id: `pdoc-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
        ownerId: req.authUser.id,
        documentLabel,
        originalName: req.file.originalname,
        mimeType: req.file.mimetype,
        fileSize: req.file.size,
        uploadedAt: new Date().toISOString(),
        storedFilename: req.file.filename,
      };

      const nextFiles = [record, ...readPersonalFiles()];
      writePersonalFiles(nextFiles);

      const { storedFilename: _s, ...publicRecord } = record;
      const userFiles = nextFiles
        .filter((f) => f.ownerId === req.authUser?.id)
        .map(({ storedFilename: _sf, ...pub }) => pub);

      res.status(201).json({ file: publicRecord, uploadedPersonalFiles: userFiles });
    } catch (error) {
      res.status(500).json({
        error: error instanceof Error ? error.message : 'Personal document upload failed.',
      });
    }
  }
);

app.get(
  '/api/user/documents/file/:fileId',
  requireAuth,
  (req: AuthedRequest, res: Response) => {
    const fileId = req.params.fileId;
    const record = readPersonalFiles().find((f) => f.id === fileId);
    if (!record) {
      res.status(404).json({ error: 'Personal document not found.' });
      return;
    }
    if (record.ownerId !== req.authUser?.id) {
      res.status(403).json({
        error: 'Forbidden: You do not have permission to access another citizen’s personal document.',
      });
      return;
    }
    const fullPath = path.join(getPersonalUploadDir(), path.basename(record.storedFilename));
    if (!fs.existsSync(fullPath)) {
      res.status(404).json({ error: 'Stored file missing on disk.' });
      return;
    }
    res.setHeader('Content-Type', record.mimeType || 'application/octet-stream');
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.setHeader(
      'Content-Disposition',
      `inline; filename="${record.originalName.replace(/[^a-zA-Z0-9._-]/g, '_')}"`
    );
    fs.createReadStream(fullPath).pipe(res);
  }
);

app.delete(
  '/api/user/documents/file/:fileId',
  requireAuth,
  (req: AuthedRequest, res: Response) => {
    const fileId = req.params.fileId;
    const all = readPersonalFiles();
    const record = all.find((f) => f.id === fileId);
    if (!record) {
      res.status(404).json({ error: 'Personal document not found.' });
      return;
    }
    if (record.ownerId !== req.authUser?.id) {
      res.status(403).json({ error: 'Forbidden: Cannot delete another user’s document.' });
      return;
    }
    const fullPath = path.join(getPersonalUploadDir(), path.basename(record.storedFilename));
    if (fs.existsSync(fullPath)) {
      fs.unlinkSync(fullPath);
    }
    const remaining = all.filter((f) => f.id !== fileId);
    writePersonalFiles(remaining);
    const userFiles = remaining
      .filter((f) => f.ownerId === req.authUser?.id)
      .map(({ storedFilename: _s, ...pub }) => pub);
    res.json({ uploadedPersonalFiles: userFiles });
  }
);

app.get('/api/documents', (req: Request, res: Response) => {
  const ownerId = resolveWorkspaceOwner(req);
  res.json({ documents: readUserWorkspace(ownerId).documents });
});

app.put('/api/documents', (req: Request, res: Response) => {
  const ownerId = resolveWorkspaceOwner(req);
  const { documents } = req.body as { documents?: CitizenDocument[] };
  if (!Array.isArray(documents)) {
    res.status(400).json({ error: 'Invalid documents payload.' });
    return;
  }
  if (ownerId === 'guest') {
    writeDocuments(documents);
  }
  writeUserWorkspace(ownerId, (curr) => ({ ...curr, documents }));
  res.json({ documents });
});

app.get('/api/saved', (req: Request, res: Response) => {
  const ownerId = resolveWorkspaceOwner(req);
  res.json({ savedIds: readUserWorkspace(ownerId).savedIds });
});

app.put('/api/saved', (req: Request, res: Response) => {
  const ownerId = resolveWorkspaceOwner(req);
  const { savedIds } = req.body as { savedIds?: string[] };
  if (!Array.isArray(savedIds)) {
    res.status(400).json({ error: 'Invalid savedIds payload.' });
    return;
  }
  if (ownerId === 'guest') {
    writeSavedIds(savedIds);
  }
  writeUserWorkspace(ownerId, (curr) => ({ ...curr, savedIds }));
  res.json({ savedIds });
});

// RAG-powered AI Assistant endpoint
app.post('/api/assistant/chat', async (req: Request, res: Response) => {
  const { prompt, history, profile } = req.body as {
    prompt?: string;
    history?: Array<{
      id: string;
      role: 'user' | 'assistant';
      content: string;
      timestamp: string;
    }>;
    profile?: EligibilityProfile;
  };

  if (!prompt || typeof prompt !== 'string' || !prompt.trim()) {
    res.status(400).json({ error: 'Prompt is required.' });
    return;
  }

  const ownerId = resolveWorkspaceOwner(req);
  const schemes = filterPublicSchemes(readSchemes());
  const documents = readUserWorkspace(ownerId).documents;
  const safeProfile: EligibilityProfile = profile || {
    age: '',
    annualIncome: '',
    state: 'Not Provided',
    district: '',
    occupation: 'Not Provided',
    educationLevel: 'Not Provided',
    farmerStatus: 'Not Provided',
    employmentStatus: 'Not Provided',
    gender: 'Not Provided',
    residence: 'Not Provided',
    socialCategory: 'Not Provided',
    isBpl: false,
    certificates: [],
  };

  const reply = await generateGroundedAssistantReply(
    prompt.trim(),
    Array.isArray(history) ? history : [],
    schemes,
    documents,
    safeProfile
  );

  res.json(reply);
});

// Admin Authentication (unified with users store & strict admin role check)
app.post('/api/admin/login', (req: Request, res: Response) => {
  const { email, password } = req.body as { email?: string; password?: string };
  const normalizedEmail = (email || '').trim().toLowerCase();
  const cleanPassword = password || '';

  const users = readUsers();
  const found = users.find((u) => u.email.toLowerCase() === normalizedEmail);
  if (!found || !verifyPassword(cleanPassword, found.passwordHash)) {
    res.status(401).json({ error: 'Invalid administrator credentials.' });
    return;
  }
  if (found.role !== 'admin') {
    res.status(403).json({
      error: 'Forbidden: This account does not have administrator privileges.',
    });
    return;
  }

  const user: AuthUser = {
    id: found.id,
    email: found.email,
    name: found.name,
    role: 'admin',
  };
  const token = signAuthToken(user);
  res.json({ token, user });
});

app.get('/api/admin/session', requireAdmin, (req: AuthedRequest, res: Response) => {
  res.json({ user: req.adminUser });
});

app.get('/api/admin/schemes', requireAdmin, (_req: AuthedRequest, res: Response) => {
  res.json({ schemes: readSchemes() });
});

app.get('/api/admin/stats', requireAdmin, (_req: AuthedRequest, res: Response) => {
  const schemes = readSchemes();
  const jobs = readPdfJobs();
  const categoryCounts = SCHEME_CATEGORIES.map((c) => ({
    category: c.name,
    count: schemes.filter((s) => s.category === c.name).length,
  }));

  const recentlyUpdatedSchemes = [...schemes]
    .sort((a, b) => b.lastUpdated.localeCompare(a.lastUpdated))
    .slice(0, 6);

  const stats: AdminDashboardStats = {
    totalSchemes: schemes.length,
    publishedSchemes: schemes.filter((s) => s.publicationStatus !== 'Draft').length,
    draftSchemes: schemes.filter((s) => s.publicationStatus === 'Draft').length,
    approvedSchemes: schemes.filter((s) => s.verificationStatus === 'Verified').length,
    demonstrationSchemes: schemes.filter((s) => s.verificationStatus === 'Demonstration').length,
    centralSchemes: schemes.filter((s) => s.level === 'Central').length,
    stateSchemes: schemes.filter((s) => s.level === 'State').length,
    categoriesCount: categoryCounts.filter((c) => c.count > 0).length,
    pendingApprovals: jobs.filter((j) => j.processingStatus === 'Pending Review').length,
    uploadedDocumentsCount: jobs.length,
    totalUploadedBytes: jobs.reduce((acc, j) => acc + (j.fileSize || 0), 0),
    schemesByCategory: categoryCounts,
    recentlyUpdatedSchemes,
  };

  res.json({ stats });
});

// Admin Manual Scheme Management
app.post('/api/admin/schemes', requireAdmin, async (req: AuthedRequest, res: Response) => {
  const body = req.body as Partial<Scheme>;
  if (!body.name?.trim() || !body.department?.trim() || !body.shortDescription?.trim()) {
    res.status(400).json({ error: 'Name, department, and short description are required.' });
    return;
  }

  const schemes = readSchemes();
  const duplicate = findDuplicateScheme(schemes, body.name, body.department);
  if (duplicate) {
    res.status(409).json({
      error: `Duplicate scheme prevented: "${duplicate.schemeName}" (${duplicate.reason})`,
      duplicate,
    });
    return;
  }

  const today = new Date().toISOString().slice(0, 10);
  const existingIds = new Set(schemes.map((s) => s.id));
  const newId = slugifySchemeId(body.name, existingIds);
  const level = body.level === 'State' ? 'State' : 'Central';
  const publicationStatus = body.publicationStatus === 'Draft' ? 'Draft' : 'Published';
  const verificationStatus = body.verificationStatus || 'Verified';

  const newScheme: Scheme = {
    id: newId,
    code:
      body.code?.trim() ||
      (level === 'Central'
        ? `IN-CEN-${String(schemes.length + 1).padStart(3, '0')}`
        : `IN-ST-${String(schemes.length + 1).padStart(3, '0')}`),
    name: body.name.trim(),
    department: body.department.trim(),
    category: body.category || 'Social Welfare',
    subcategory: body.subcategory?.trim() || undefined,
    purpose: body.purpose?.trim() || undefined,
    level,
    stateName: level === 'State' ? body.stateName?.trim() || 'State Government' : undefined,
    shortDescription: body.shortDescription.trim(),
    overview: (body.overview || body.shortDescription).trim(),
    benefits:
      Array.isArray(body.benefits) && body.benefits.length > 0
        ? body.benefits
        : ['Added manually via Admin Scheme Management.'],
    eligibilityConditions:
      Array.isArray(body.eligibilityConditions) && body.eligibilityConditions.length > 0
        ? body.eligibilityConditions
        : ['Verify official eligibility rules.'],
    requiredDocuments:
      Array.isArray(body.requiredDocuments) && body.requiredDocuments.length > 0
        ? body.requiredDocuments
        : ['Aadhaar Card'],
    applicationProcedure:
      Array.isArray(body.applicationProcedure) && body.applicationProcedure.length > 0
        ? body.applicationProcedure
        : ['Submit application on the official government portal.'],
    restrictionsAndExceptions:
      Array.isArray(body.restrictionsAndExceptions) &&
      body.restrictionsAndExceptions.length > 0
        ? body.restrictionsAndExceptions
        : ['Refer to official portal guidelines for exclusions.'],
    officialWebsite: body.officialWebsite?.trim() || 'https://www.india.gov.in',
    sourceDocument: body.sourceDocument || {
      fileName: 'manual-admin-entry',
      uploadedAt: today,
      version: 1,
      sourceLabel: `Entered manually by ${req.adminUser?.email}`,
    },
    createdDate: today,
    lastUpdated: today,
    verifiedAt: verificationStatus === 'Verified' ? today : undefined,
    verificationStatus,
    publicationStatus,
    version: 1,
    versionHistory: [
      {
        version: 1,
        updatedAt: today,
        updatedBy: req.adminUser?.email || 'Admin',
        changeSummary: `Created manually (${publicationStatus}).`,
      },
    ],
    isPopular: Boolean(body.isPopular),
    isRecent: true,
    criteria: body.criteria || {
      genders: ['All'],
      residence: ['All'],
      states: level === 'State' && body.stateName?.trim() ? [body.stateName.trim()] : undefined,
    },
  };

  const updated = [newScheme, ...schemes];
  writeSchemes(updated);
  if (publicationStatus === 'Published') {
    appendNotification({
      type: 'new_scheme',
      title: `New Scheme Added: ${newScheme.name}`,
      message: `${newScheme.name} has been published under ${newScheme.category} (${newScheme.level}).`,
      schemeId: newScheme.id,
      schemeName: newScheme.name,
      category: newScheme.category,
    });
  }
  await syncRagIndex(filterPublicSchemes(updated), getGeminiClient());
  res.status(201).json({ scheme: newScheme, schemes: updated });
});

app.put('/api/admin/schemes/:id', requireAdmin, async (req: AuthedRequest, res: Response) => {
  const schemeId = req.params.id;
  const updates = req.body as Partial<Scheme> & { changeSummary?: string };
  const schemes = readSchemes();
  const idx = schemes.findIndex((s) => s.id === schemeId);

  if (idx === -1) {
    res.status(404).json({ error: 'Scheme not found.' });
    return;
  }

  const current = schemes[idx];
  const candidateName = updates.name ?? current.name;
  const candidateDept = updates.department ?? current.department;

  const duplicate = findDuplicateScheme(
    schemes,
    candidateName,
    candidateDept,
    undefined,
    schemeId
  );
  if (duplicate) {
    res.status(409).json({
      error: `Duplicate conflict with "${duplicate.schemeName}": ${duplicate.reason}`,
      duplicate,
    });
    return;
  }

  const today = new Date().toISOString().slice(0, 10);
  const nextVersion = (current.version || 1) + 1;
  const nextVerification =
    updates.verificationStatus ||
    (current.verificationStatus === 'Demonstration' ? 'Verified' : current.verificationStatus);
  const updatedScheme: Scheme = {
    ...current,
    ...updates,
    id: current.id,
    createdDate: current.createdDate || current.lastUpdated,
    lastUpdated: today,
    verifiedAt:
      nextVerification === 'Verified'
        ? current.verifiedAt || today
        : current.verifiedAt,
    version: nextVersion,
    verificationStatus: nextVerification,
    publicationStatus: updates.publicationStatus ?? current.publicationStatus ?? 'Published',
    versionHistory: [
      ...(current.versionHistory || []),
      {
        version: nextVersion,
        updatedAt: today,
        updatedBy: req.adminUser?.email || 'Admin',
        changeSummary: updates.changeSummary || 'Updated scheme fields via Admin workspace.',
      },
    ],
  };

  const nextSchemes = [...schemes];
  nextSchemes[idx] = updatedScheme;
  writeSchemes(nextSchemes);
  if (updatedScheme.publicationStatus !== 'Draft') {
    appendNotification({
      type: 'updated_scheme',
      title: `Updated Scheme (v${nextVersion}): ${updatedScheme.name}`,
      message:
        updates.changeSummary ||
        `Scheme conditions or guidelines for ${updatedScheme.name} were updated.`,
      schemeId: updatedScheme.id,
      schemeName: updatedScheme.name,
      category: updatedScheme.category,
    });
  }
  await syncRagIndex(filterPublicSchemes(nextSchemes), getGeminiClient());
  res.json({ scheme: updatedScheme, schemes: nextSchemes });
});

app.delete('/api/admin/schemes/:id', requireAdmin, async (req: AuthedRequest, res: Response) => {
  const schemeId = req.params.id;
  const schemes = readSchemes();
  const filtered = schemes.filter((s) => s.id !== schemeId);
  if (filtered.length === schemes.length) {
    res.status(404).json({ error: 'Scheme not found.' });
    return;
  }
  writeSchemes(filtered);
  await syncRagIndex(filterPublicSchemes(filtered), getGeminiClient());

  const savedIds = readSavedIds().filter((id) => id !== schemeId);
  writeSavedIds(savedIds);

  res.json({ schemes: filtered, savedIds });
});

app.post('/api/admin/schemes/reset', requireAdmin, async (_req: AuthedRequest, res: Response) => {
  writeSchemes(INITIAL_SCHEMES);
  await syncRagIndex(filterPublicSchemes(INITIAL_SCHEMES), getGeminiClient());
  res.json({ schemes: INITIAL_SCHEMES });
});

// Admin PDF Upload, Async Multi-Document Batch Processing & Consolidation Workflow
function updateAdminJobRecord(
  jobId: string,
  updater: (current: PdfExtractionJob) => PdfExtractionJob
): PdfExtractionJob | null {
  const jobs = readPdfJobs();
  const idx = jobs.findIndex((j) => j.id === jobId);
  if (idx === -1) return null;
  const updated = updater(jobs[idx]);
  jobs[idx] = updated;
  writePdfJobs(jobs);
  return updated;
}

async function processAdminPdfJobAsync(
  jobId: string,
  filePath: string,
  fileName: string,
  targetSchemeId?: string
): Promise<void> {
  try {
    updateAdminJobRecord(jobId, (curr) => ({
      ...curr,
      jobStatus: 'Extracting Text',
      stageDetail: 'Reading PDF pages and extracting text...',
    }));

    const fileBuffer = fs.readFileSync(filePath);
    let extractedPages: string[] = [];
    let extractedText = '';
    try {
      const pdfResult = await extractText(new Uint8Array(fileBuffer), {
        mergePages: false,
      });
      extractedPages = Array.isArray(pdfResult.text)
        ? pdfResult.text.map((p) => String(p))
        : [String(pdfResult.text || '')];
      extractedText = extractedPages
        .map((p, idx) => `--- Page ${idx + 1} ---\n${p}`)
        .join('\n\n');
    } catch (err) {
      extractedText = '';
      console.error('Admin PDF text extraction warning:', err);
    }

    const selectableChars = extractedPages.reduce((sum, p) => sum + p.trim().length, 0);
    const requiresOcr = selectableChars < 150;

    updateAdminJobRecord(jobId, (curr) => ({
      ...curr,
      jobStatus: 'Analyzing with Gemini',
      stageDetail: requiresOcr
        ? `Running visual OCR & extraction across ${extractedPages.length} page(s)...`
        : `Analyzing extracted text (${extractedPages.length} page(s)) with Gemini...`,
      pageCount: extractedPages.length,
      requiresOcr,
    }));

    const aiResult = await extractSchemeFromPdf(fileBuffer, extractedText, fileName);

    updateAdminJobRecord(jobId, (curr) => ({
      ...curr,
      jobStatus: 'Generating Eligibility Criteria',
      stageDetail: 'Structuring eligibility conditions, benefits, and page citations...',
    }));

    const currentJob = readPdfJobs().find((j) => j.id === jobId);
    const sha256 = currentJob?.sha256 || computeSha256(fileBuffer);
    const schemes = readSchemes();
    const duplicateMatch = targetSchemeId
      ? null
      : findDuplicateScheme(schemes, aiResult.draft.name, aiResult.draft.department, sha256);

    updateAdminJobRecord(jobId, (curr) => ({
      ...curr,
      processingStatus: 'Pending Review',
      jobStatus: 'Completed',
      stageDetail: `Completed extraction across ${extractedPages.length} page(s). Pending Admin review.`,
      pageCount: extractedPages.length,
      requiresOcr,
      publicationOrRevisionDate: aiResult.publicationOrRevisionDate,
      importantDates: aiResult.importantDates,
      pageCitations: aiResult.pageCitations,
      extractionMethod: aiResult.extractionMethod,
      extractionError: aiResult.extractionError,
      extractedTextPreview: extractedText.slice(0, 6000),
      extractedPages,
      duplicateMatch,
      draft: aiResult.draft,
    }));
  } catch (err) {
    const msg = sanitizeErrorText(err instanceof Error ? err.message : String(err));
    updateAdminJobRecord(jobId, (curr) => ({
      ...curr,
      processingStatus: 'Extraction Failed',
      jobStatus: 'Failed',
      stageDetail: msg || 'Background extraction failed.',
      extractionError: msg || 'Background extraction failed.',
    }));
  }
}

function createSkeletonAdminDraft(fileName: string): ExtractedSchemeDraft {
  return {
    name: fileName.replace(/\.pdf$/i, '').replace(/[_-]+/g, ' ').trim() || 'Uploaded Scheme PDF',
    department: 'Extracting...',
    category: 'Social Welfare',
    level: 'Central',
    shortDescription: 'Extraction in progress...',
    overview: 'Extraction in progress...',
    benefits: [],
    eligibilityConditions: [],
    requiredDocuments: [],
    applicationProcedure: [],
    restrictionsAndExceptions: [],
    officialWebsite: 'https://www.india.gov.in',
    criteria: { genders: ['All'], residence: ['All'] },
  };
}

app.get('/api/admin/pdf/jobs', requireAdmin, (_req: AuthedRequest, res: Response) => {
  res.json({ jobs: readPdfJobs() });
});

app.get('/api/admin/pdf/groups/:groupId', requireAdmin, (req: AuthedRequest, res: Response) => {
  const groupId = (req.params.groupId || '').trim();
  const jobs = readPdfJobs();
  const groupJobs = jobs.filter((j) => j.groupId === groupId);
  const consolidated = consolidateAdminGroup(groupId, jobs);
  res.json({ groupId, jobs: groupJobs, consolidated });
});

app.post(
  '/api/admin/pdf/upload',
  requireAdmin,
  upload.single('file'),
  async (req: AuthedRequest, res: Response) => {
    try {
      if (!req.file) {
        res.status(400).json({ error: 'Please select a PDF file to upload.' });
        return;
      }

      const targetSchemeId =
        typeof req.body?.targetSchemeId === 'string' && req.body.targetSchemeId.trim()
          ? req.body.targetSchemeId.trim()
          : undefined;
      const groupId =
        typeof req.body?.groupId === 'string' && req.body.groupId.trim()
          ? req.body.groupId.trim()
          : `grp_${Date.now()}`;

      const fileBuffer = fs.readFileSync(req.file.path);
      if (!hasValidPdfSignature(fileBuffer)) {
        fs.unlinkSync(req.file.path);
        res.status(400).json({
          error: 'Invalid PDF file signature (%PDF- header missing). Upload rejected.',
        });
        return;
      }
      const sha256 = computeSha256(fileBuffer);

      let extractedPages: string[] = [];
      let extractedText = '';
      try {
        const pdfResult = await extractText(new Uint8Array(fileBuffer), {
          mergePages: false,
        });
        extractedPages = Array.isArray(pdfResult.text)
          ? pdfResult.text.map((p) => String(p))
          : [String(pdfResult.text || '')];
        extractedText = extractedPages
          .map((p, idx) => `--- Page ${idx + 1} ---\n${p}`)
          .join('\n\n');
      } catch (err) {
        extractedText = '';
        console.error('PDF text extraction warning:', err);
      }

      const aiResult = await extractSchemeFromPdf(
        fileBuffer,
        extractedText,
        req.file.originalname
      );

      const schemes = readSchemes();
      const duplicateMatch = targetSchemeId
        ? null
        : findDuplicateScheme(
            schemes,
            aiResult.draft.name,
            aiResult.draft.department,
            sha256
          );

      const job: PdfExtractionJob = {
        id: `pdf_${Date.now()}`,
        groupId,
        documentId: `adoc_${Date.now()}_${crypto.randomBytes(2).toString('hex')}`,
        documentVersion: 1,
        fileName: req.file.originalname,
        fileSize: req.file.size,
        storageUrl: `/api/pdfs/${req.file.filename}`,
        sha256,
        uploadedAt: new Date().toISOString(),
        uploadedBy: req.adminUser?.email || 'Admin',
        processingStatus: 'Pending Review',
        jobStatus: 'Completed',
        stageDetail: `Completed extraction across ${extractedPages.length} page(s).`,
        pageCount: extractedPages.length,
        publicationOrRevisionDate: aiResult.publicationOrRevisionDate,
        importantDates: aiResult.importantDates,
        pageCitations: aiResult.pageCitations,
        extractionMethod: aiResult.extractionMethod,
        extractionError: aiResult.extractionError,
        extractedTextPreview: extractedText.slice(0, 6000),
        extractedPages,
        targetSchemeId,
        duplicateMatch,
        draft: aiResult.draft,
      };

      const jobs = [job, ...readPdfJobs()];
      writePdfJobs(jobs);
      res.status(201).json({ job, jobs });
    } catch (error) {
      const msg = error instanceof Error ? error.message : 'PDF upload failed.';
      res.status(500).json({ error: msg });
    }
  }
);

app.post(
  '/api/admin/pdf/upload-batch',
  requireAdmin,
  upload.fields([
    { name: 'files', maxCount: 10 },
    { name: 'file', maxCount: 1 },
  ]),
  (req: AuthedRequest, res: Response) => {
    try {
      const fileGroups = (req.files || {}) as Record<string, Express.Multer.File[]>;
      const uploadedFiles = [...(fileGroups.files || []), ...(fileGroups.file || [])];
      if (uploadedFiles.length === 0) {
        res.status(400).json({ error: 'Please select one or more PDF files to upload.' });
        return;
      }

      const targetSchemeId =
        typeof req.body?.targetSchemeId === 'string' && req.body.targetSchemeId.trim()
          ? req.body.targetSchemeId.trim()
          : undefined;
      const groupId =
        typeof req.body?.groupId === 'string' && req.body.groupId.trim()
          ? req.body.groupId.trim()
          : `grp_${Date.now()}_${crypto.randomBytes(2).toString('hex')}`;

      const existingJobs = readPdfJobs();
      const groupActiveJobs = existingJobs.filter(
        (j) => j.groupId === groupId && j.processingStatus !== 'Rejected'
      );
      const groupShaSet = new Map<string, string>();
      for (const j of groupActiveJobs) {
        if (j.sha256) groupShaSet.set(j.sha256, j.fileName);
      }

      const createdJobs: PdfExtractionJob[] = [];
      const duplicates: Array<{ fileName: string; sha256: string; duplicateOf: string }> = [];
      const rejectedFiles: Array<{ fileName: string; message: string }> = [];

      for (const file of uploadedFiles) {
        const buffer = fs.readFileSync(file.path);
        if (file.size === 0 || !hasValidPdfSignature(buffer)) {
          fs.unlinkSync(file.path);
          rejectedFiles.push({
            fileName: file.originalname,
            message: `"${file.originalname}" is empty or not a valid PDF (%PDF- header missing).`,
          });
          continue;
        }

        const sha256 = computeSha256(buffer);
        const dupName = groupShaSet.get(sha256);
        if (dupName) {
          fs.unlinkSync(file.path);
          duplicates.push({
            fileName: file.originalname,
            sha256,
            duplicateOf: dupName,
          });
          continue;
        }
        groupShaSet.set(sha256, file.originalname);

        const jobId = `pdf_${Date.now()}_${crypto.randomBytes(3).toString('hex')}`;
        const job: PdfExtractionJob = {
          id: jobId,
          groupId,
          documentId: `adoc_${Date.now()}_${crypto.randomBytes(3).toString('hex')}`,
          documentVersion: 1,
          fileName: file.originalname,
          fileSize: file.size,
          storageUrl: `/api/pdfs/${file.filename}`,
          sha256,
          uploadedAt: new Date().toISOString(),
          uploadedBy: req.adminUser?.email || 'Admin',
          processingStatus: 'Pending Review',
          jobStatus: 'Uploaded',
          stageDetail: 'Uploaded to server; queued for text extraction...',
          extractionMethod: 'gemini',
          extractedTextPreview: '',
          targetSchemeId,
          duplicateMatch: null,
          draft: createSkeletonAdminDraft(file.originalname),
        };
        createdJobs.push(job);
      }

      const nextAllJobs = [...createdJobs, ...existingJobs];
      writePdfJobs(nextAllJobs);

      for (let i = 0; i < createdJobs.length; i++) {
        const job = createdJobs[i];
        const diskFile = uploadedFiles.find((f) => `/api/pdfs/${f.filename}` === job.storageUrl);
        if (diskFile) {
          void processAdminPdfJobAsync(job.id, diskFile.path, job.fileName, targetSchemeId);
        }
      }

      res.status(202).json({
        groupId,
        jobs: createdJobs,
        allJobs: nextAllJobs,
        duplicates,
        rejectedFiles,
      });
    } catch (error) {
      const msg = error instanceof Error ? error.message : 'Batch PDF upload failed.';
      res.status(500).json({ error: msg });
    }
  }
);

app.post(
  '/api/admin/pdf/:jobId/replace',
  requireAdmin,
  upload.single('file'),
  (req: AuthedRequest, res: Response) => {
    try {
      if (!req.file) {
        res.status(400).json({ error: 'Replacement PDF file is required.' });
        return;
      }
      const { jobId } = req.params;
      const jobs = readPdfJobs();
      const idx = jobs.findIndex((j) => j.id === jobId);
      if (idx === -1) {
        fs.unlinkSync(req.file.path);
        res.status(404).json({ error: 'Document job not found.' });
        return;
      }

      const buffer = fs.readFileSync(req.file.path);
      if (req.file.size === 0 || !hasValidPdfSignature(buffer)) {
        fs.unlinkSync(req.file.path);
        res.status(422).json({ error: 'Replacement file is not a valid PDF document.' });
        return;
      }
      const sha256 = computeSha256(buffer);
      const current = jobs[idx];

      // Prevent duplicate with another document in same group
      if (current.groupId) {
        const dup = jobs.find(
          (j) =>
            j.id !== jobId &&
            j.groupId === current.groupId &&
            j.sha256 === sha256 &&
            j.processingStatus !== 'Rejected'
        );
        if (dup) {
          fs.unlinkSync(req.file.path);
          res.status(409).json({
            error: `Duplicate document: matches "${dup.fileName}" already attached to this scheme group.`,
          });
          return;
        }
      }

      // Clean up old file if unreferenced elsewhere
      const oldFilename = path.basename(current.storageUrl);
      const stillUsed = jobs.some(
        (j) => j.id !== jobId && path.basename(j.storageUrl) === oldFilename
      );
      if (!stillUsed) {
        const oldPath = path.join(getUploadDir(), oldFilename);
        if (fs.existsSync(oldPath)) fs.unlinkSync(oldPath);
      }

      const updated: PdfExtractionJob = {
        ...current,
        fileName: req.file.originalname,
        fileSize: req.file.size,
        storageUrl: `/api/pdfs/${req.file.filename}`,
        sha256,
        uploadedAt: new Date().toISOString(),
        uploadedBy: req.adminUser?.email || 'Admin',
        documentVersion: (current.documentVersion || 1) + 1,
        processingStatus: 'Pending Review',
        jobStatus: 'Uploaded',
        stageDetail: 'Replacement PDF uploaded; queued for extraction...',
        extractionError: undefined,
      };
      jobs[idx] = updated;
      writePdfJobs(jobs);

      void processAdminPdfJobAsync(
        jobId,
        req.file.path,
        req.file.originalname,
        updated.targetSchemeId
      );

      res.status(202).json({ job: updated, jobs });
    } catch (error) {
      res.status(500).json({
        error: error instanceof Error ? error.message : 'Replacement upload failed.',
      });
    }
  }
);

app.post('/api/admin/pdf/:jobId/reanalyze', requireAdmin, (req: AuthedRequest, res: Response) => {
  const { jobId } = req.params;
  const jobs = readPdfJobs();
  const idx = jobs.findIndex((j) => j.id === jobId);
  if (idx === -1) {
    res.status(404).json({ error: 'Document job not found.' });
    return;
  }
  const current = jobs[idx];
  const fullPath = path.join(getUploadDir(), path.basename(current.storageUrl));
  if (!fs.existsSync(fullPath)) {
    res.status(404).json({ error: 'Stored PDF file missing on server.' });
    return;
  }
  const updated: PdfExtractionJob = {
    ...current,
    processingStatus: 'Pending Review',
    jobStatus: 'Uploaded',
    stageDetail: 'Queued for re-analysis...',
    extractionError: undefined,
  };
  jobs[idx] = updated;
  writePdfJobs(jobs);

  void processAdminPdfJobAsync(jobId, fullPath, updated.fileName, updated.targetSchemeId);
  res.status(202).json({ job: updated, jobs });
});

app.delete('/api/admin/pdf/:jobId', requireAdmin, (req: AuthedRequest, res: Response) => {
  const { jobId } = req.params;
  const jobs = readPdfJobs();
  const target = jobs.find((j) => j.id === jobId);
  if (!target) {
    res.status(404).json({ error: 'Document job not found.' });
    return;
  }
  const remaining = jobs.filter((j) => j.id !== jobId);
  writePdfJobs(remaining);

  const fileBase = path.basename(target.storageUrl);
  const referencedByJob = remaining.some((j) => path.basename(j.storageUrl) === fileBase);
  const referencedByScheme = readSchemes().some(
    (s) =>
      path.basename(s.sourceDocument?.storageUrl || '') === fileBase ||
      (s.documents || []).some((d) => path.basename(d.storageUrl || '') === fileBase)
  );
  if (!referencedByJob && !referencedByScheme) {
    const diskPath = path.join(getUploadDir(), fileBase);
    if (fs.existsSync(diskPath)) fs.unlinkSync(diskPath);
  }

  res.json({ deletedJobId: jobId, jobs: remaining });
});

app.put('/api/admin/pdf/:jobId', requireAdmin, (req: AuthedRequest, res: Response) => {
  const { jobId } = req.params;
  const { draft, targetSchemeId } = req.body as {
    draft?: ExtractedSchemeDraft;
    targetSchemeId?: string | null;
  };
  const jobs = readPdfJobs();
  const idx = jobs.findIndex((j) => j.id === jobId);
  if (idx === -1) {
    res.status(404).json({ error: 'Extraction job not found.' });
    return;
  }

  const current = jobs[idx];
  const nextDraft = draft || current.draft;
  const resolvedTarget =
    targetSchemeId === null ? undefined : targetSchemeId ?? current.targetSchemeId;
  const schemes = readSchemes();
  const duplicateMatch = resolvedTarget
    ? null
    : findDuplicateScheme(schemes, nextDraft.name, nextDraft.department, current.sha256);

  const updatedJob: PdfExtractionJob = {
    ...current,
    draft: nextDraft,
    targetSchemeId: resolvedTarget,
    duplicateMatch,
  };

  jobs[idx] = updatedJob;
  writePdfJobs(jobs);
  res.json({ job: updatedJob, jobs });
});

app.post('/api/admin/pdf/approve-group', requireAdmin, async (req: AuthedRequest, res: Response) => {
  const { groupId, draft, targetSchemeId, approvedJobIds } = req.body as {
    groupId?: string;
    draft?: ExtractedSchemeDraft;
    targetSchemeId?: string;
    approvedJobIds?: string[];
  };

  if (!groupId || !groupId.trim()) {
    res.status(400).json({ error: 'groupId is required.' });
    return;
  }

  const jobs = readPdfJobs();
  const groupJobs = jobs.filter(
    (j) =>
      j.groupId === groupId &&
      j.processingStatus !== 'Rejected' &&
      j.processingStatus !== 'Extraction Failed' &&
      j.jobStatus !== 'Failed' &&
      (!approvedJobIds || approvedJobIds.includes(j.id))
  );

  if (groupJobs.length === 0) {
    res.status(400).json({
      error: 'No successfully analyzed documents were found in this group to approve.',
    });
    return;
  }

  const pendingOrProcessing = groupJobs.some(
    (j) =>
      j.jobStatus &&
      j.jobStatus !== 'Completed'
  );
  if (pendingOrProcessing) {
    res.status(409).json({
      error: 'One or more documents in this group are still processing. Wait for completion or remove failed files before approving.',
    });
    return;
  }

  const consolidated = consolidateAdminGroup(groupId, groupJobs);
  const finalDraft = draft || consolidated?.draft;
  if (!finalDraft || !finalDraft.name?.trim() || !finalDraft.department?.trim()) {
    res.status(400).json({ error: 'Scheme name and department are required to approve.' });
    return;
  }

  const schemes = readSchemes();
  const resolvedTargetId = targetSchemeId || consolidated?.targetSchemeId;
  const today = new Date().toISOString().slice(0, 10);

  const newAttachedDocs: SchemeAttachedDocument[] = groupJobs.map((j) => ({
    documentId: j.documentId || j.id,
    fileName: j.fileName,
    fileSize: j.fileSize,
    storageUrl: j.storageUrl,
    uploadedAt: j.uploadedAt.slice(0, 10),
    uploadedBy: j.uploadedBy,
    version: j.documentVersion || 1,
    sha256: j.sha256,
    pageCount: j.pageCount || j.extractedPages?.length,
    publicationOrRevisionDate: j.publicationOrRevisionDate,
    sourceLabel: `Admin-approved PDF (${j.documentVersion ? `doc v${j.documentVersion}` : 'verified'})`,
    pageReferences: j.extractedPages?.length ? `Pages 1–${j.extractedPages.length}` : undefined,
  }));

  const docNamesLabel = newAttachedDocs.map((d) => d.fileName).join(', ');
  let approvedScheme: Scheme;
  let nextSchemes: Scheme[];

  if (resolvedTargetId) {
    const existingIdx = schemes.findIndex((s) => s.id === resolvedTargetId);
    if (existingIdx === -1) {
      res.status(404).json({ error: 'Target scheme for version update was not found.' });
      return;
    }

    const existing = schemes[existingIdx];
    const nextVersion = (existing.version || 1) + 1;
    const mergedDocsMap = new Map<string, SchemeAttachedDocument>();
    for (const oldDoc of existing.documents || []) {
      mergedDocsMap.set(oldDoc.sha256 || oldDoc.documentId, oldDoc);
    }
    for (const nd of newAttachedDocs) {
      mergedDocsMap.set(nd.sha256 || nd.documentId, nd);
    }
    const mergedDocs = Array.from(mergedDocsMap.values());

    approvedScheme = {
      ...existing,
      name: finalDraft.name.trim(),
      department: finalDraft.department.trim(),
      category: finalDraft.category,
      purpose: finalDraft.purpose ?? existing.purpose,
      level: finalDraft.level,
      stateName: finalDraft.level === 'State' ? finalDraft.stateName : undefined,
      shortDescription: finalDraft.shortDescription.trim(),
      overview: finalDraft.overview.trim(),
      benefits: finalDraft.benefits,
      eligibilityConditions: finalDraft.eligibilityConditions,
      requiredDocuments: finalDraft.requiredDocuments,
      applicationProcedure: finalDraft.applicationProcedure,
      restrictionsAndExceptions: finalDraft.restrictionsAndExceptions,
      officialWebsite: finalDraft.officialWebsite || existing.officialWebsite,
      criteria: finalDraft.criteria || existing.criteria,
      sourceDocument: {
        documentId: mergedDocs[mergedDocs.length - 1]?.documentId,
        fileName: mergedDocs[mergedDocs.length - 1]?.fileName || docNamesLabel,
        storageUrl: mergedDocs[mergedDocs.length - 1]?.storageUrl,
        uploadedAt: today,
        version: nextVersion,
        sha256: mergedDocs[mergedDocs.length - 1]?.sha256,
        sourceLabel: `Admin-approved multi-document update (${mergedDocs.length} PDFs, v${nextVersion})`,
        pageReferences: consolidated?.sourcePageReferences.join('; '),
      },
      documents: mergedDocs,
      conflicts: consolidated?.conflicts || [],
      sourcePageReferences: consolidated?.sourcePageReferences || [],
      lastUpdated: today,
      verifiedAt: today,
      verificationStatus: 'Verified',
      publicationStatus: 'Published',
      version: nextVersion,
      isRecent: true,
      versionHistory: [
        ...(existing.versionHistory || []),
        {
          version: nextVersion,
          updatedAt: today,
          updatedBy: req.adminUser?.email || 'Admin',
          changeSummary: `Updated from ${newAttachedDocs.length} approved PDF(s): ${docNamesLabel}`,
          sourceDocumentName: docNamesLabel,
        },
      ],
    };

    nextSchemes = [...schemes];
    nextSchemes[existingIdx] = approvedScheme;
  } else {
    const duplicate = findDuplicateScheme(
      schemes,
      finalDraft.name,
      finalDraft.department,
      newAttachedDocs[0]?.sha256
    );
    if (duplicate) {
      res.status(409).json({
        error: `Duplicate scheme blocked: "${duplicate.schemeName}" (${duplicate.reason}). Link this group as an update to the existing scheme instead.`,
        duplicate,
      });
      return;
    }

    const existingIds = new Set(schemes.map((s) => s.id));
    const newId = slugifySchemeId(finalDraft.name, existingIds);

    approvedScheme = {
      id: newId,
      code:
        finalDraft.level === 'Central'
          ? `IN-CEN-${String(schemes.length + 1).padStart(3, '0')}`
          : `IN-ST-${String(schemes.length + 1).padStart(3, '0')}`,
      name: finalDraft.name.trim(),
      department: finalDraft.department.trim(),
      category: finalDraft.category,
      purpose: finalDraft.purpose,
      level: finalDraft.level,
      stateName: finalDraft.level === 'State' ? finalDraft.stateName : undefined,
      shortDescription: finalDraft.shortDescription.trim(),
      overview: finalDraft.overview.trim(),
      benefits: finalDraft.benefits,
      eligibilityConditions: finalDraft.eligibilityConditions,
      requiredDocuments: finalDraft.requiredDocuments,
      applicationProcedure: finalDraft.applicationProcedure,
      restrictionsAndExceptions: finalDraft.restrictionsAndExceptions,
      officialWebsite: finalDraft.officialWebsite || 'https://www.india.gov.in',
      sourceDocument: {
        documentId: newAttachedDocs[0]?.documentId,
        fileName: newAttachedDocs[0]?.fileName || docNamesLabel,
        storageUrl: newAttachedDocs[0]?.storageUrl,
        uploadedAt: today,
        version: 1,
        sha256: newAttachedDocs[0]?.sha256,
        sourceLabel:
          newAttachedDocs.length > 1
            ? `Consolidated from ${newAttachedDocs.length} verified PDFs`
            : 'Admin-approved PDF upload',
        pageReferences: consolidated?.sourcePageReferences.join('; '),
      },
      documents: newAttachedDocs,
      conflicts: consolidated?.conflicts || [],
      sourcePageReferences: consolidated?.sourcePageReferences || [],
      createdDate: today,
      lastUpdated: today,
      verifiedAt: today,
      verificationStatus: 'Verified',
      publicationStatus: 'Published',
      version: 1,
      isRecent: true,
      versionHistory: [
        {
          version: 1,
          updatedAt: today,
          updatedBy: req.adminUser?.email || 'Admin',
          changeSummary: `Created and verified from ${newAttachedDocs.length} PDF document(s): ${docNamesLabel}`,
          sourceDocumentName: docNamesLabel,
        },
      ],
      criteria: finalDraft.criteria || {
        genders: ['All'],
        residence: ['All'],
      },
    };

    nextSchemes = [approvedScheme, ...schemes];
  }

  writeSchemes(nextSchemes);

  const approvedSet = new Set(groupJobs.map((j) => j.id));
  const nowIso = new Date().toISOString();
  const nextJobs = jobs.map((j) =>
    approvedSet.has(j.id)
      ? {
          ...j,
          targetSchemeId: approvedScheme.id,
          duplicateMatch: null,
          processingStatus: 'Approved' as const,
          reviewedAt: nowIso,
          reviewedBy: req.adminUser?.email || 'Admin',
          approvedSchemeId: approvedScheme.id,
        }
      : j
  );
  writePdfJobs(nextJobs);

  appendNotification({
    type: resolvedTargetId ? 'updated_scheme' : 'new_scheme',
    title: resolvedTargetId
      ? `Scheme Updated (v${approvedScheme.version}): ${approvedScheme.name}`
      : `New Verified Scheme: ${approvedScheme.name}`,
    message: resolvedTargetId
      ? `Updated from ${newAttachedDocs.length} official PDF(s) in ${approvedScheme.category}.`
      : `Published from ${newAttachedDocs.length} verified PDF(s) under ${approvedScheme.category}.`,
    schemeId: approvedScheme.id,
    schemeName: approvedScheme.name,
    category: approvedScheme.category,
  });

  await syncRagIndex(filterPublicSchemes(nextSchemes), getGeminiClient(), nextJobs);

  res.json({
    scheme: approvedScheme,
    schemes: nextSchemes,
    jobs: nextJobs,
  });
});

app.post('/api/admin/pdf/:jobId/approve', requireAdmin, async (req: AuthedRequest, res: Response) => {
  const { jobId } = req.params;
  const { draft, targetSchemeId } = req.body as {
    draft?: ExtractedSchemeDraft;
    targetSchemeId?: string;
  };

  const jobs = readPdfJobs();
  const idx = jobs.findIndex((j) => j.id === jobId);
  if (idx === -1) {
    res.status(404).json({ error: 'Extraction job not found.' });
    return;
  }

  const job = jobs[idx];
  if (job.processingStatus === 'Approved') {
    res.status(400).json({ error: 'This extraction job has already been approved.' });
    return;
  }

  const finalDraft = draft || job.draft;
  if (!finalDraft.name?.trim() || !finalDraft.department?.trim()) {
    res.status(400).json({ error: 'Scheme name and department are required to approve.' });
    return;
  }

  const schemes = readSchemes();
  const resolvedTargetId = targetSchemeId || job.targetSchemeId;
  const today = new Date().toISOString().slice(0, 10);
  const docEntry: SchemeAttachedDocument = {
    documentId: job.documentId || job.id,
    fileName: job.fileName,
    fileSize: job.fileSize,
    storageUrl: job.storageUrl,
    uploadedAt: today,
    uploadedBy: job.uploadedBy,
    version: job.documentVersion || 1,
    sha256: job.sha256,
    pageCount: job.pageCount || job.extractedPages?.length,
    publicationOrRevisionDate: job.publicationOrRevisionDate,
    sourceLabel: 'Admin-approved PDF',
    pageReferences: job.extractedPages?.length
      ? `Pages 1–${job.extractedPages.length}`
      : undefined,
  };

  let approvedScheme: Scheme;
  let nextSchemes: Scheme[];

  if (resolvedTargetId) {
    const existingIdx = schemes.findIndex((s) => s.id === resolvedTargetId);
    if (existingIdx === -1) {
      res.status(404).json({ error: 'Target scheme for version update was not found.' });
      return;
    }

    const existing = schemes[existingIdx];
    const nextVersion = (existing.version || 1) + 1;
    const prevDocs = existing.documents || [];
    const mergedDocs = [
      ...prevDocs.filter((d) => d.sha256 !== docEntry.sha256),
      docEntry,
    ];

    approvedScheme = {
      ...existing,
      name: finalDraft.name.trim(),
      department: finalDraft.department.trim(),
      category: finalDraft.category,
      purpose: finalDraft.purpose ?? existing.purpose,
      level: finalDraft.level,
      stateName: finalDraft.level === 'State' ? finalDraft.stateName : undefined,
      shortDescription: finalDraft.shortDescription.trim(),
      overview: finalDraft.overview.trim(),
      benefits: finalDraft.benefits,
      eligibilityConditions: finalDraft.eligibilityConditions,
      requiredDocuments: finalDraft.requiredDocuments,
      applicationProcedure: finalDraft.applicationProcedure,
      restrictionsAndExceptions: finalDraft.restrictionsAndExceptions,
      officialWebsite: finalDraft.officialWebsite || existing.officialWebsite,
      criteria: finalDraft.criteria || existing.criteria,
      sourceDocument: {
        documentId: docEntry.documentId,
        fileName: job.fileName,
        storageUrl: job.storageUrl,
        uploadedAt: today,
        version: nextVersion,
        sha256: job.sha256,
        sourceLabel: `Admin-approved PDF v${nextVersion}`,
      },
      documents: mergedDocs,
      lastUpdated: today,
      verifiedAt: today,
      verificationStatus: 'Verified',
      publicationStatus: 'Published',
      version: nextVersion,
      isRecent: true,
      versionHistory: [
        ...(existing.versionHistory || []),
        {
          version: nextVersion,
          updatedAt: today,
          updatedBy: req.adminUser?.email || 'Admin',
          changeSummary: `Updated from approved PDF "${job.fileName}"`,
          sourceDocumentName: job.fileName,
        },
      ],
    };

    nextSchemes = [...schemes];
    nextSchemes[existingIdx] = approvedScheme;
  } else {
    const duplicate = findDuplicateScheme(
      schemes,
      finalDraft.name,
      finalDraft.department,
      job.sha256
    );
    if (duplicate) {
      res.status(409).json({
        error: `Duplicate scheme blocked: "${duplicate.schemeName}" (${duplicate.reason}). Link this PDF as an update to the existing scheme instead.`,
        duplicate,
      });
      return;
    }

    const existingIds = new Set(schemes.map((s) => s.id));
    const newId = slugifySchemeId(finalDraft.name, existingIds);

    approvedScheme = {
      id: newId,
      code:
        finalDraft.level === 'Central'
          ? `IN-CEN-${String(schemes.length + 1).padStart(3, '0')}`
          : `IN-ST-${String(schemes.length + 1).padStart(3, '0')}`,
      name: finalDraft.name.trim(),
      department: finalDraft.department.trim(),
      category: finalDraft.category,
      purpose: finalDraft.purpose,
      level: finalDraft.level,
      stateName: finalDraft.level === 'State' ? finalDraft.stateName : undefined,
      shortDescription: finalDraft.shortDescription.trim(),
      overview: finalDraft.overview.trim(),
      benefits: finalDraft.benefits,
      eligibilityConditions: finalDraft.eligibilityConditions,
      requiredDocuments: finalDraft.requiredDocuments,
      applicationProcedure: finalDraft.applicationProcedure,
      restrictionsAndExceptions: finalDraft.restrictionsAndExceptions,
      officialWebsite: finalDraft.officialWebsite || 'https://www.india.gov.in',
      sourceDocument: {
        documentId: docEntry.documentId,
        fileName: job.fileName,
        storageUrl: job.storageUrl,
        uploadedAt: today,
        version: 1,
        sha256: job.sha256,
        sourceLabel: 'Admin-approved PDF upload',
        pageReferences: job.extractedPages?.length
          ? `Pages 1–${job.extractedPages.length}`
          : undefined,
      },
      documents: [docEntry],
      createdDate: today,
      lastUpdated: today,
      verifiedAt: today,
      verificationStatus: 'Verified',
      publicationStatus: 'Published',
      version: 1,
      isRecent: true,
      versionHistory: [
        {
          version: 1,
          updatedAt: today,
          updatedBy: req.adminUser?.email || 'Admin',
          changeSummary: `Created and verified from PDF "${job.fileName}"`,
          sourceDocumentName: job.fileName,
        },
      ],
      criteria: finalDraft.criteria || {
        genders: ['All'],
        residence: ['All'],
      },
    };

    nextSchemes = [approvedScheme, ...schemes];
  }

  writeSchemes(nextSchemes);

  jobs[idx] = {
    ...job,
    draft: finalDraft,
    targetSchemeId: resolvedTargetId,
    duplicateMatch: null,
    processingStatus: 'Approved',
    reviewedAt: new Date().toISOString(),
    reviewedBy: req.adminUser?.email || 'Admin',
    approvedSchemeId: approvedScheme.id,
  };
  writePdfJobs(jobs);

  appendNotification({
    type: resolvedTargetId ? 'updated_scheme' : 'new_scheme',
    title: resolvedTargetId
      ? `Scheme Updated (v${approvedScheme.version}): ${approvedScheme.name}`
      : `New Verified Scheme: ${approvedScheme.name}`,
    message: resolvedTargetId
      ? `Updated from official PDF "${job.fileName}" in ${approvedScheme.category}.`
      : `Published from verified PDF "${job.fileName}" under ${approvedScheme.category}.`,
    schemeId: approvedScheme.id,
    schemeName: approvedScheme.name,
    category: approvedScheme.category,
  });

  await syncRagIndex(filterPublicSchemes(nextSchemes), getGeminiClient(), jobs);

  res.json({
    scheme: approvedScheme,
    schemes: nextSchemes,
    job: jobs[idx],
    jobs,
  });
});

app.post('/api/admin/pdf/:jobId/reject', requireAdmin, (req: AuthedRequest, res: Response) => {
  const { jobId } = req.params;
  const jobs = readPdfJobs();
  const idx = jobs.findIndex((j) => j.id === jobId);
  if (idx === -1) {
    res.status(404).json({ error: 'Extraction job not found.' });
    return;
  }

  jobs[idx] = {
    ...jobs[idx],
    processingStatus: 'Rejected',
    reviewedAt: new Date().toISOString(),
    reviewedBy: req.adminUser?.email || 'Admin',
  };
  writePdfJobs(jobs);
  res.json({ job: jobs[idx], jobs });
});

// Multer & API error handler
app.use((err: unknown, _req: Request, res: Response, next: NextFunction) => {
  if (err instanceof multer.MulterError) {
    if (err.code === 'LIMIT_FILE_SIZE') {
      res.status(413).json({
        error:
          'Uploaded file exceeds the maximum allowed size (10 MB for public PDFs, 15 MB for admin PDFs).',
        code: 'FILE_TOO_LARGE',
      });
      return;
    }
    res.status(400).json({ error: err.message, code: err.code });
    return;
  }
  if (err instanceof Error) {
    const custom = err as Error & { statusCode?: number; code?: string };
    const status = typeof custom.statusCode === 'number' ? custom.statusCode : 400;
    res.status(status).json({
      error: sanitizeErrorText(err.message),
      code: custom.code || 'REQUEST_ERROR',
    });
    return;
  }
  next();
});

async function startServer(): Promise<void> {
  if (process.env.NODE_ENV !== 'production') {
    process.env.DISABLE_HMR ||= 'true';
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: {
        middlewareMode: true,
        hmr: false,
        ws: false,
      },
      appType: 'spa',
      plugins: [
        {
          name: 'yojana-disable-hmr-websocket-client',
          enforce: 'post',
          transform(code, id) {
            const cleanId = id.split('?')[0].replace(/\\/g, '/');
            if (cleanId.endsWith('/vite/dist/client/client.mjs')) {
              const patched = code.replace(
                /let wsTransport = createWebSocketModuleRunnerTransport\(\{[\s\S]*?\}\);/,
                'let wsTransport = { async connect() {}, async disconnect() {}, send() {} };'
              );
              if (patched !== code) {
                return { code: patched, map: null };
              }
            }
          },
        },
      ],
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(process.cwd(), 'dist');
    app.use(express.static(distDir));
    app.get('/{*splat}', (_req: Request, res: Response) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  }

  const server = app.listen(PORT, '0.0.0.0', () => {
    console.log(`YojanaAI server running on http://localhost:${PORT}`);
    void syncRagIndex(filterPublicSchemes(readSchemes()), getGeminiClient()).catch(
      (err) => {
        console.warn(
          'Initial RAG index warmup warning:',
          err instanceof Error ? err.message : String(err)
        );
      }
    );
  });
  server.requestTimeout = 120000;
  server.headersTimeout = 125000;
  server.keepAliveTimeout = 65000;
}

startServer();
