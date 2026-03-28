require('dotenv').config();
const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const path = require('path');
const mysql = require('mysql2/promise');
const jwt = require('jsonwebtoken');

const app = express();
const PORT = process.env.PORT || 3000;

// ----------------------
// Middleware
// ----------------------
app.use(cors());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// Serve all static files (JS, CSS, images, background images)
app.use(express.static(path.join(__dirname, '../public')));

// JWT secret
const JWT_SECRET = process.env.JWT_SECRET || 'your_super_secret_jwt_key';

// Superadmin credentials (server-side only)
const SUPERADMIN_USERNAME = process.env.SUPERADMIN_USERNAME || 'adminuser';
const SUPERADMIN_PASSWORD = process.env.SUPERADMIN_PASSWORD || 'password';

// MySQL connection pool
const db = mysql.createPool({
  host: process.env.DB_HOST || 'localhost',
  user: process.env.DB_USER || 'root',
  password: process.env.DB_PASSWORD || '',
  database: process.env.DB_NAME || 'Sky_Dock',
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0
});

// Test DB connection
(async () => {
  try {
    const conn = await db.getConnection();
    console.log('✅ MySQL connected successfully');
    conn.release();
  } catch (err) {
    console.error('❌ DB Connection Error:', err.message);
  }
})();

// ----------------------
// Authentication Middleware
// ----------------------
function authMiddleware(req, res, next) {
  const token = req.headers['authorization']?.split(' ')[1];
  if (!token) return res.status(401).json({ success: false, message: 'Token required' });

  try {
    const decoded = jwt.verify(token, JWT_SECRET);
    req.user = decoded;
    next();
  } catch (err) {
    return res.status(403).json({ success: false, message: 'Invalid or expired token' });
  }
}

// ----------------------
// Routes
// ----------------------

// Serve login page
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, '../public/login.html'));
});

// Serve dashboard page (protected)
app.get('/dashboard.html', authMiddleware, (req, res) => {
  res.sendFile(path.join(__dirname, '../public/dashboard.html'));
});

// Login route
app.post('/api/login', async (req, res) => {
  const { username, password } = req.body;

  if (!username || !password) {
    return res.status(400).json({ success: false, message: 'Username and password required' });
  }

  // Superadmin login
  if (username === SUPERADMIN_USERNAME && password === SUPERADMIN_PASSWORD) {
    const token = jwt.sign({ username, role: 'superadmin' }, JWT_SECRET, { expiresIn: '12h' });
    return res.json({ success: true, token });
  }

  // Future: check regular users from DB
  // const [rows] = await db.query('SELECT * FROM users WHERE username = ? AND password = ?', [username, password]);
  // if (rows.length > 0) { ... }

  return res.status(401).json({ success: false, message: 'Invalid username or password' });
});

// Example protected API route
app.get('/api/dashboard', authMiddleware, (req, res) => {
  res.json({ success: true, message: 'Welcome to the dashboard', user: req.user });
});

// ----------------------
// Start server
// ----------------------
app.listen(PORT, () => {
  console.log(`🚀 Server running at http://localhost:${PORT}`);
});

// Logout button
const logoutBtn = document.getElementById("logoutBtn");

logoutBtn.addEventListener("click", async () => {
  try {
    // Optional: inform server to invalidate token if you implement token blacklisting
    // await fetch('/api/logout', {
    //   method: 'POST',
    //   headers: { 'Authorization': 'Bearer ' + token }
    // });

    // Remove JWT from localStorage
    localStorage.removeItem("token");

    // Redirect to login page
    window.location.href = "login.html";
  } catch (err) {
    console.error("Logout failed:", err);
  }
});

// Prevent caching for protected routes
app.use((req, res, next) => {
  res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, private');
  next();
});