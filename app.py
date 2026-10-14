"""
╔══════════════════════════════════════════════════════════════════╗
║         ChefMate AI – Intelligent Recipe Generator               ║
║  Flask + Groq API + Bootstrap 5  |  Single-file application      ║
╚══════════════════════════════════════════════════════════════════╝

Features:
  1. Nutrition & Recipe RAG / Knowledge Agent
  2. Personalized Recipe & Substitution Agent

Run:
  python app.py
  → http://127.0.0.1:5000
"""

import os
import secrets
from functools import wraps
from flask import Flask, render_template_string, request, jsonify, session, redirect, url_for, flash
from groq import Groq
from dotenv import load_dotenv

# ── Load environment variables (.env preferred, .env.example as fallback) ──
load_dotenv(dotenv_path=".env", override=False)          # try .env first
load_dotenv(dotenv_path=".env.example", override=False)  # fallback if .env absent

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")  # free open-source model

app = Flask(__name__)
# Secret key for session signing – reads from env or generates a random one
app.secret_key = os.getenv("SECRET_KEY", secrets.token_hex(24))

# ── User store: pre-seeded accounts + any dynamically registered users ──
# Key = username (lowercase), Value = plaintext password.
# In production replace this with a real database and hashed passwords.
users_db = {
    "chef":  "123",
    "admin": "chefmate",
}

# ══════════════════════════════════════════════════════════════════
#  GROQ HELPER
# ══════════════════════════════════════════════════════════════════

def generate_response(system_prompt: str, user_message: str, max_tokens: int = 1024) -> str:
    """
    Send a prompt to the Groq LLM and return the text response.
    Includes basic input validation and error handling.
    """
    # Input validation
    if not GROQ_API_KEY:
        return "⚠️ GROQ_API_KEY is not set. Please add it to your .env file."
    if not user_message or not user_message.strip():
        return "⚠️ Please enter a valid message."

    try:
        client = Groq(api_key=GROQ_API_KEY)
        chat_completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system",  "content": system_prompt},
                {"role": "user",    "content": user_message.strip()},
            ],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return chat_completion.choices[0].message.content or ""

    except Exception as e:
        return f"⚠️ Groq API error: {str(e)}"


# ══════════════════════════════════════════════════════════════════
#  SYSTEM PROMPTS FOR EACH AGENT
# ══════════════════════════════════════════════════════════════════

KNOWLEDGE_AGENT_PROMPT = """
You are ChefMate AI, an expert culinary assistant and nutritionist.
Your knowledge spans both local Indian cuisine and global dishes.

Your role is to:
- Answer cooking and recipe-related questions clearly and helpfully.
- Explain food combinations, flavour pairings, and cooking techniques.
- Provide practical cooking tips with examples from Indian and global cuisine.
- Share nutritional information and health benefits of ingredients.
- Always be friendly, encouraging, and beginner-friendly.

Format your answers with clear headings, bullet points, and emoji where appropriate.
Keep responses concise but complete.
"""

RECIPE_AGENT_PROMPT = """
You are ChefMate AI, a creative and expert recipe generator.
A user will give you:
  - Ingredients they have at home
  - Dietary preferences (e.g., vegan, low-carb, gluten-free, dairy-free)
  - Cooking time constraints

Generate a complete, structured recipe using EXACTLY this Markdown format:

### 🍳 Adapted Recipe Title
[A creative, engaging name for the dish]

### 📝 Step-by-Step Cooking Instructions
1. [Step 1]
2. [Step 2]
3. [Step 3]
(continue as needed)

### 🔄 Ingredient Substitutions Made
* **Original Ingredient:** [X] → **Substituted With:** [Y]  *(Reason: ...)*
(list only substitutions actually made; write "None needed" if no substitutions)

### 📊 Estimated Nutritional Facts (Per Serving)
* **Calories:** ~[kcal]
* **Protein:** [X]g
* **Carbs:** [Y]g
* **Fats:** [Z]g

### 🛒 Shopping List (Missing Items)
* [item 1]
* [item 2]
(list only items NOT mentioned by the user; write "You have everything!" if nothing is missing)

Be practical, accurate with nutrition estimates, and make the dish delicious.
"""

# ══════════════════════════════════════════════════════════════════
#  BASE HTML TEMPLATE  (shared layout with sidebar)
# ══════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════
#  LOGIN PAGE TEMPLATE  (standalone – no sidebar)
# ══════════════════════════════════════════════════════════════════

# Shared CSS + head used by both LOGIN_HTML and REGISTER_HTML
_AUTH_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>ChefMate AI – {page_title}</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet"/>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css" rel="stylesheet"/>
  <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet"/>
  <style>
    :root {{ --primary: #e85d04; --primary-d: #c44b02; }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: 'Inter', sans-serif;
      min-height: 100vh;
      display: flex; align-items: center; justify-content: center;
      background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
    }}
    .auth-card {{
      width: 100%; max-width: 430px;
      background: #fff; border-radius: 18px;
      box-shadow: 0 20px 60px rgba(0,0,0,.35); overflow: hidden;
    }}
    .auth-header {{
      background: linear-gradient(135deg, var(--primary), var(--primary-d));
      padding: 32px 32px 24px; text-align: center; color: #fff;
    }}
    .auth-header h1 {{
      font-family: 'Playfair Display', serif; font-size: 1.8rem; margin: 8px 0 4px;
    }}
    .auth-header p {{ opacity: .85; margin: 0; font-size: .9rem; }}
    /* Tab switcher */
    .auth-tabs {{
      display: flex; border-bottom: 2px solid #f0f0f0;
    }}
    .auth-tab {{
      flex: 1; padding: 13px 0; text-align: center;
      font-weight: 600; font-size: .9rem; cursor: pointer;
      color: #999; border-bottom: 3px solid transparent;
      margin-bottom: -2px; transition: all .2s; user-select: none;
    }}
    .auth-tab.active {{ color: var(--primary); border-bottom-color: var(--primary); }}
    .auth-body {{ padding: 28px 32px 32px; }}
    .form-control {{
      border-radius: 8px !important; border: 1.5px solid #dee2e6 !important;
      font-size: .93rem !important; padding: 10px 14px !important;
    }}
    .form-control:focus {{
      border-color: var(--primary) !important;
      box-shadow: 0 0 0 3px rgba(232,93,4,.15) !important;
    }}
    .btn-auth {{
      background: var(--primary); color: #fff; border: none;
      border-radius: 8px; padding: 11px; font-weight: 600;
      font-size: 1rem; width: 100%; transition: background .2s;
    }}
    .btn-auth:hover {{ background: var(--primary-d); color: #fff; }}
    .input-group-text {{
      background: #f8f9fa;
      border-radius: 8px 0 0 8px !important;
      border: 1.5px solid #dee2e6 !important; border-right: none !important;
      color: #6c757d;
    }}
    .input-group .form-control {{ border-radius: 0 8px 8px 0 !important; }}
    .hint-box {{
      background: #fff8f0; border: 1px solid #ffd8b3; border-radius: 8px;
      padding: 10px 14px; font-size: .8rem; color: #7a4a1e; margin-top: 14px;
    }}
    .switch-link {{
      text-align: center; margin-top: 18px; font-size: .85rem; color: #6c757d;
    }}
    .switch-link a {{ color: var(--primary); font-weight: 600; text-decoration: none; }}
    .switch-link a:hover {{ text-decoration: underline; }}
    /* Password strength bar */
    .strength-bar {{
      height: 4px; border-radius: 4px; margin-top: 6px;
      transition: width .3s, background .3s;
    }}
    .strength-label {{ font-size: .75rem; color: #888; margin-top: 3px; }}
  </style>
</head>"""

LOGIN_HTML = _AUTH_HEAD.format(page_title="Login") + """
<body>
  <div class="auth-card">
    <div class="auth-header">
      <div style="font-size:2.6rem;">🍽️</div>
      <h1>ChefMate AI</h1>
      <p>Your intelligent culinary kitchen</p>
    </div>

    <!-- Tab row -->
    <div class="auth-tabs">
      <div class="auth-tab active" onclick="window.location='/login'">
        <i class="bi bi-box-arrow-in-right me-1"></i>Sign In
      </div>
      <div class="auth-tab" onclick="window.location='/register'">
        <i class="bi bi-person-plus me-1"></i>Sign Up
      </div>
    </div>

    <div class="auth-body">

      <!-- Success message (after registration redirect) -->
      {% if success %}
      <div class="alert alert-success alert-dismissible d-flex align-items-center gap-2 py-2 px-3 mb-3">
        <i class="bi bi-check-circle-fill"></i>
        <span>{{ success }}</span>
        <button type="button" class="btn-close ms-auto" data-bs-dismiss="alert"></button>
      </div>
      {% endif %}

      <!-- Error message -->
      {% if error %}
      <div class="alert alert-danger alert-dismissible d-flex align-items-center gap-2 py-2 px-3 mb-3">
        <i class="bi bi-exclamation-triangle-fill"></i>
        <span>{{ error }}</span>
        <button type="button" class="btn-close ms-auto" data-bs-dismiss="alert"></button>
      </div>
      {% endif %}

      <form method="POST" action="/login" autocomplete="on">
        <div class="mb-3">
          <label for="username" class="form-label fw-semibold" style="font-size:.9rem;">Username</label>
          <div class="input-group">
            <span class="input-group-text"><i class="bi bi-person-fill"></i></span>
            <input type="text" id="username" name="username" class="form-control"
                   placeholder="Enter username" value="{{ username or '' }}"
                   required autofocus autocomplete="username"/>
          </div>
        </div>

        <div class="mb-4">
          <label for="password" class="form-label fw-semibold" style="font-size:.9rem;">Password</label>
          <div class="input-group">
            <span class="input-group-text"><i class="bi bi-lock-fill"></i></span>
            <input type="password" id="password" name="password" class="form-control"
                   placeholder="Enter password" required autocomplete="current-password"/>
            <button class="btn btn-outline-secondary" type="button" id="togglePwd"
                    style="border-radius:0 8px 8px 0;border:1.5px solid #dee2e6;border-left:none;">
              <i class="bi bi-eye" id="togglePwdIcon"></i>
            </button>
          </div>
        </div>

        <button type="submit" class="btn-auth">
          <i class="bi bi-box-arrow-in-right me-2"></i>Sign In
        </button>
      </form>

      <div class="hint-box">
        <strong>Demo credentials:</strong>&nbsp;
        Username: <code>chef</code> &nbsp;|&nbsp; Password: <code>123</code>
      </div>

      <div class="switch-link">
        Don't have an account? <a href="/register">Sign up here</a>
      </div>
    </div>
  </div>

  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
  <script>
    document.getElementById('togglePwd').addEventListener('click', function () {
      const pwd  = document.getElementById('password');
      const icon = document.getElementById('togglePwdIcon');
      pwd.type = pwd.type === 'password' ? 'text' : 'password';
      icon.className = pwd.type === 'password' ? 'bi bi-eye' : 'bi bi-eye-slash';
    });
  </script>
</body>
</html>
"""

# ── Registration page ─────────────────────────────────────────────
REGISTER_HTML = _AUTH_HEAD.format(page_title="Register") + """
<body>
  <div class="auth-card">
    <div class="auth-header">
      <div style="font-size:2.6rem;">🍽️</div>
      <h1>ChefMate AI</h1>
      <p>Create your culinary account</p>
    </div>

    <!-- Tab row -->
    <div class="auth-tabs">
      <div class="auth-tab" onclick="window.location='/login'">
        <i class="bi bi-box-arrow-in-right me-1"></i>Sign In
      </div>
      <div class="auth-tab active" onclick="window.location='/register'">
        <i class="bi bi-person-plus me-1"></i>Sign Up
      </div>
    </div>

    <div class="auth-body">

      <!-- Error message -->
      {% if error %}
      <div class="alert alert-danger alert-dismissible d-flex align-items-center gap-2 py-2 px-3 mb-3">
        <i class="bi bi-exclamation-triangle-fill"></i>
        <span>{{ error }}</span>
        <button type="button" class="btn-close ms-auto" data-bs-dismiss="alert"></button>
      </div>
      {% endif %}

      <form method="POST" action="/register" autocomplete="off" id="regForm">
        <!-- Username -->
        <div class="mb-3">
          <label for="reg_username" class="form-label fw-semibold" style="font-size:.9rem;">
            Username <span class="text-muted fw-normal">(min. 3 chars, letters/numbers only)</span>
          </label>
          <div class="input-group">
            <span class="input-group-text"><i class="bi bi-person-fill"></i></span>
            <input type="text" id="reg_username" name="username" class="form-control"
                   placeholder="Choose a username" value="{{ username or '' }}"
                   required minlength="3" maxlength="30" pattern="[A-Za-z0-9_]+"
                   title="Letters, numbers and underscores only" autocomplete="off"/>
          </div>
        </div>

        <!-- Password -->
        <div class="mb-3">
          <label for="reg_password" class="form-label fw-semibold" style="font-size:.9rem;">
            Password <span class="text-muted fw-normal">(min. 4 chars)</span>
          </label>
          <div class="input-group">
            <span class="input-group-text"><i class="bi bi-lock-fill"></i></span>
            <input type="password" id="reg_password" name="password" class="form-control"
                   placeholder="Create a password" required minlength="4"
                   autocomplete="new-password" oninput="checkStrength(this.value)"/>
            <button class="btn btn-outline-secondary" type="button" id="toggleRegPwd"
                    style="border-radius:0 8px 8px 0;border:1.5px solid #dee2e6;border-left:none;">
              <i class="bi bi-eye" id="toggleRegPwdIcon"></i>
            </button>
          </div>
          <!-- Strength bar -->
          <div id="strengthBar" class="strength-bar" style="width:0;background:#dee2e6;"></div>
          <div id="strengthLabel" class="strength-label"></div>
        </div>

        <!-- Confirm Password -->
        <div class="mb-4">
          <label for="reg_confirm" class="form-label fw-semibold" style="font-size:.9rem;">Confirm Password</label>
          <div class="input-group">
            <span class="input-group-text"><i class="bi bi-shield-lock-fill"></i></span>
            <input type="password" id="reg_confirm" name="confirm" class="form-control"
                   placeholder="Repeat your password" required autocomplete="new-password"/>
          </div>
          <div id="confirmMsg" style="font-size:.78rem;margin-top:4px;"></div>
        </div>

        <button type="submit" class="btn-auth" id="regBtn">
          <i class="bi bi-person-check me-2"></i>Create Account
        </button>
      </form>

      <div class="switch-link">
        Already have an account? <a href="/login">Sign in here</a>
      </div>
    </div>
  </div>

  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
  <script>
    // ── Password visibility toggle ────────────────────────────────
    document.getElementById('toggleRegPwd').addEventListener('click', function () {
      const pwd  = document.getElementById('reg_password');
      const icon = document.getElementById('toggleRegPwdIcon');
      pwd.type = pwd.type === 'password' ? 'text' : 'password';
      icon.className = pwd.type === 'password' ? 'bi bi-eye' : 'bi bi-eye-slash';
    });

    // ── Password strength indicator ───────────────────────────────
    function checkStrength(val) {
      const bar   = document.getElementById('strengthBar');
      const label = document.getElementById('strengthLabel');
      let score = 0;
      if (val.length >= 4)  score++;
      if (val.length >= 8)  score++;
      if (/[A-Z]/.test(val)) score++;
      if (/[0-9]/.test(val)) score++;
      if (/[^A-Za-z0-9]/.test(val)) score++;
      const levels = [
        { pct:'20%', color:'#e74c3c', text:'Very weak'  },
        { pct:'40%', color:'#e67e22', text:'Weak'        },
        { pct:'60%', color:'#f1c40f', text:'Fair'        },
        { pct:'80%', color:'#2ecc71', text:'Strong'      },
        { pct:'100%',color:'#27ae60', text:'Very strong' },
      ];
      const lv = levels[Math.min(score, 4)];
      bar.style.width      = val ? lv.pct   : '0';
      bar.style.background = val ? lv.color : '#dee2e6';
      label.textContent    = val ? lv.text  : '';
      label.style.color    = lv.color;
    }

    // ── Confirm-password live match check ─────────────────────────
    document.getElementById('reg_confirm').addEventListener('input', function () {
      const pwd = document.getElementById('reg_password').value;
      const msg = document.getElementById('confirmMsg');
      if (!this.value) { msg.textContent = ''; return; }
      if (this.value === pwd) {
        msg.innerHTML = '<span style="color:#27ae60"><i class="bi bi-check-circle me-1"></i>Passwords match</span>';
      } else {
        msg.innerHTML = '<span style="color:#e74c3c"><i class="bi bi-x-circle me-1"></i>Passwords do not match</span>';
      }
    });

    // ── Client-side confirm match before submit ───────────────────
    document.getElementById('regForm').addEventListener('submit', function (e) {
      const pwd     = document.getElementById('reg_password').value;
      const confirm = document.getElementById('reg_confirm').value;
      if (pwd !== confirm) {
        e.preventDefault();
        document.getElementById('confirmMsg').innerHTML =
          '<span style="color:#e74c3c"><i class="bi bi-x-circle me-1"></i>Passwords do not match</span>';
        document.getElementById('reg_confirm').focus();
      }
    });
  </script>
</body>
</html>
"""

# ══════════════════════════════════════════════════════════════════
#  BASE HTML TEMPLATE  (shared layout with sidebar)
# ══════════════════════════════════════════════════════════════════

BASE_HTML = """<!DOCTYPE html>
<html lang="en" id="htmlRoot">
<head>
  <!-- Apply saved theme BEFORE first paint to prevent flash -->
  <script>
    (function(){
      var t = localStorage.getItem('chefmate_theme') || 'light';
      document.documentElement.setAttribute('data-bs-theme', t);
    })();
  </script>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>ChefMate AI – {{ page_title }}</title>

  <!-- Bootstrap 5 -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet"/>
  <!-- Bootstrap Icons -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css" rel="stylesheet"/>
  <!-- Google Fonts -->
  <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet"/>
  <!-- Marked.js – Markdown renderer -->
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>

  <style>
    :root {
      --primary:   #e85d04;
      --primary-d: #c44b02;
      --secondary: #ffd166;
      --bg:        #fffbf5;
      --sidebar:   #1a1a2e;
      --sidebar-a: #e85d04;
      --card-bg:   #ffffff;
      --text:      #2d2d2d;
      --muted:     #6c757d;
      --radius:    14px;
    }
    * { box-sizing: border-box; }
    body {
      font-family: 'Inter', sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
    }

    /* ── Sidebar ─────────────────────────────────────────────── */
    #sidebar {
      width: 260px;
      min-height: 100vh;
      background: var(--sidebar);
      color: #fff;
      display: flex;
      flex-direction: column;
      position: fixed;
      top: 0; left: 0; bottom: 0;
      z-index: 1000;
      transition: transform .3s ease;
    }
    #sidebar .brand {
      padding: 28px 20px 20px;
      border-bottom: 1px solid rgba(255,255,255,.08);
    }
    #sidebar .brand h4 {
      font-family: 'Playfair Display', serif;
      color: var(--secondary);
      margin: 0;
      font-size: 1.3rem;
    }
    #sidebar .brand small { color: rgba(255,255,255,.5); font-size:.75rem; }
    #sidebar nav { padding: 16px 0; flex: 1; }
    #sidebar nav a {
      display: flex; align-items: center; gap: 12px;
      padding: 12px 24px;
      color: rgba(255,255,255,.75);
      text-decoration: none;
      font-size: .92rem;
      font-weight: 500;
      transition: all .2s;
    }
    #sidebar nav a:hover,
    #sidebar nav a.active {
      background: rgba(232,93,4,.18);
      color: #fff;
      border-left: 3px solid var(--primary);
    }
    #sidebar nav a i { font-size: 1.1rem; }
    #sidebar .sidebar-footer {
      padding: 16px 20px;
      border-top: 1px solid rgba(255,255,255,.08);
      font-size: .75rem;
      color: rgba(255,255,255,.35);
    }

    /* ── Main content ─────────────────────────────────────────── */
    #main {
      margin-left: 260px;
      flex: 1;
      padding: 32px 36px;
      max-width: calc(100vw - 260px);
    }

    /* ── Topbar ───────────────────────────────────────────────── */
    .topbar {
      display: flex; align-items: center; gap: 12px;
      margin-bottom: 28px;
    }
    .topbar h2 {
      font-family: 'Playfair Display', serif;
      color: var(--primary);
      margin: 0;
    }

    /* ── Cards ────────────────────────────────────────────────── */
    .card {
      background: var(--card-bg);
      border: none;
      border-radius: var(--radius);
      box-shadow: 0 2px 12px rgba(0,0,0,.07);
    }
    .card-header-custom {
      background: linear-gradient(135deg, var(--primary), var(--primary-d));
      color: #fff;
      border-radius: var(--radius) var(--radius) 0 0;
      padding: 16px 20px;
      font-weight: 600;
    }

    /* ── Chat / response area ─────────────────────────────────── */
    .response-box {
      background: #f8f9fa;
      border: 1px solid #e9ecef;
      border-radius: 10px;
      padding: 20px 24px;
      min-height: 80px;
      font-size: .93rem;
      line-height: 1.7;
    }
    .response-box h3, .response-box h4 { color: var(--primary); }
    .response-box ul, .response-box ol { padding-left: 1.4rem; }

    /* ── Buttons ──────────────────────────────────────────────── */
    .btn-chef {
      background: var(--primary);
      color: #fff;
      border: none;
      border-radius: 8px;
      padding: 10px 24px;
      font-weight: 600;
      transition: background .2s;
    }
    .btn-chef:hover { background: var(--primary-d); color:#fff; }

    /* ── Spinner overlay ──────────────────────────────────────── */
    #loadingOverlay {
      display: none;
      position: fixed; inset: 0;
      background: rgba(0,0,0,.45);
      z-index: 9999;
      align-items: center; justify-content: center;
      flex-direction: column; gap: 14px; color: #fff;
      font-size: 1.1rem;
    }
    #loadingOverlay.show { display: flex; }

    /* ── Hero card (home) ─────────────────────────────────────── */
    .hero-card {
      background: linear-gradient(135deg, var(--primary) 0%, #f48c06 100%);
      color: #fff;
      border-radius: var(--radius);
      padding: 40px 36px;
      margin-bottom: 28px;
    }
    .hero-card h1 { font-family: 'Playfair Display', serif; font-size: 2.2rem; }
    .hero-card p  { opacity: .9; font-size: 1.05rem; }

    /* ── Feature cards (home) ─────────────────────────────────── */
    .feature-card {
      background: var(--card-bg);
      border-radius: var(--radius);
      padding: 24px;
      height: 100%;
      box-shadow: 0 2px 12px rgba(0,0,0,.07);
      transition: transform .2s, box-shadow .2s;
    }
    .feature-card:hover {
      transform: translateY(-4px);
      box-shadow: 0 8px 24px rgba(0,0,0,.12);
    }
    .feature-icon {
      width: 52px; height: 52px;
      border-radius: 12px;
      display: flex; align-items: center; justify-content: center;
      font-size: 1.5rem;
      margin-bottom: 14px;
    }

    /* ── Textarea / form ──────────────────────────────────────── */
    textarea, input[type=text], select {
      border-radius: 8px !important;
      border: 1.5px solid #dee2e6 !important;
      font-size: .93rem !important;
    }
    textarea:focus, input:focus, select:focus {
      border-color: var(--primary) !important;
      box-shadow: 0 0 0 3px rgba(232,93,4,.15) !important;
    }

    /* ── Badge pills ──────────────────────────────────────────── */
    .diet-badge {
      display: inline-block;
      background: rgba(232,93,4,.1);
      color: var(--primary);
      border-radius: 20px;
      padding: 3px 11px;
      font-size: .8rem;
      font-weight: 600;
      margin: 3px 2px;
      cursor: pointer;
      border: 1.5px solid transparent;
      transition: all .15s;
    }
    .diet-badge:hover, .diet-badge.selected {
      background: var(--primary);
      color: #fff;
    }

    /* ── Responsive ───────────────────────────────────────────── */
    @media (max-width: 768px) {
      #sidebar { transform: translateX(-100%); }
      #sidebar.open { transform: translateX(0); }
      #main { margin-left: 0; padding: 20px 16px; max-width: 100vw; }
      .topbar .menu-toggle { display: block !important; }
    }

    /* ── Tip chips (About) ────────────────────────────────────── */
    .tip-chip {
      background: #fff3cd; border: 1px solid #ffc107;
      border-radius: 8px; padding: 10px 14px;
      font-size: .88rem; margin-bottom: 8px;
    }

    /* ── Theme toggle button ──────────────────────────────────── */
    #themeToggle {
      width: 36px; height: 36px;
      border-radius: 50%;
      border: 1.5px solid rgba(255,255,255,.25);
      background: rgba(255,255,255,.12);
      color: var(--secondary);
      display: flex; align-items: center; justify-content: center;
      cursor: pointer; font-size: 1rem;
      transition: background .2s, border-color .2s;
      flex-shrink: 0;
    }
    #themeToggle:hover {
      background: rgba(255,255,255,.22);
      border-color: rgba(255,255,255,.5);
    }

    /* ── Dark mode overrides ──────────────────────────────────── */
    [data-bs-theme="dark"] {
      --bg:      #0f0f1a;
      --card-bg: #1c1c2e;
      --text:    #e8e8f0;
      --muted:   #9a9ab0;
    }
    [data-bs-theme="dark"] body {
      background: var(--bg);
      color: var(--text);
    }
    [data-bs-theme="dark"] #main {
      background: var(--bg);
    }
    [data-bs-theme="dark"] .card {
      background: var(--card-bg);
      box-shadow: 0 2px 12px rgba(0,0,0,.3);
    }
    [data-bs-theme="dark"] .feature-card {
      background: var(--card-bg);
    }
    [data-bs-theme="dark"] .response-box {
      background: #13131f;
      border-color: #2a2a3e;
      color: var(--text);
    }
    [data-bs-theme="dark"] .tip-chip {
      background: #2a2504; border-color: #7a6000; color: #ffd166;
    }
    [data-bs-theme="dark"] textarea,
    [data-bs-theme="dark"] input[type=text],
    [data-bs-theme="dark"] input[type=number],
    [data-bs-theme="dark"] select {
      background: #1c1c2e !important;
      color: var(--text) !important;
      border-color: #3a3a52 !important;
    }
    [data-bs-theme="dark"] .form-control,
    [data-bs-theme="dark"] .form-select {
      background: #1c1c2e;
      color: var(--text);
      border-color: #3a3a52;
    }
    [data-bs-theme="dark"] .table {
      color: var(--text);
    }
    [data-bs-theme="dark"] .table-light thead {
      background: #2a2a3e;
      color: var(--text);
    }
    [data-bs-theme="dark"] .list-group-item {
      background: var(--card-bg);
      color: var(--text);
      border-color: #2a2a3e;
    }
    [data-bs-theme="dark"] .list-group-item-action:hover {
      background: rgba(232,93,4,.12) !important;
    }
    [data-bs-theme="dark"] .diet-badge {
      background: rgba(232,93,4,.15);
    }
    [data-bs-theme="dark"] .hint-box,
    [data-bs-theme="dark"] .alert-info {
      background: #1a1505; border-color: #5a4500; color: #ffd166;
    }
  </style>
</head>
<body>

<!-- ══ Loading Overlay ══════════════════════════════════════════ -->
<div id="loadingOverlay">
  <div class="spinner-border text-warning" style="width:3rem;height:3rem;" role="status"></div>
  <span>ChefMate AI is cooking your response…</span>
</div>

<!-- ══ Sidebar ══════════════════════════════════════════════════ -->
<div id="sidebar">
  <div class="brand">
    <h4>🍽️ ChefMate AI</h4>
    <small>Intelligent Recipe Generator</small>
  </div>
  <nav>
    <a href="/"           class="{{ 'active' if active=='home'     else '' }}"><i class="bi bi-house-fill"></i> Home</a>
    <a href="/chat"       class="{{ 'active' if active=='chat'     else '' }}"><i class="bi bi-chat-dots-fill"></i> Recipe Chat</a>
    <a href="/pantry"     class="{{ 'active' if active=='pantry'   else '' }}"><i class="bi bi-basket3-fill"></i> Pantry Helper</a>
    <a href="/about"      class="{{ 'active' if active=='about'    else '' }}"><i class="bi bi-info-circle-fill"></i> About</a>
  </nav>
  <!-- Welcome + Logout -->
  <div class="px-4 py-3" style="border-top:1px solid rgba(255,255,255,.08);">
    <div class="d-flex align-items-center gap-2 mb-2">
      <i class="bi bi-person-circle" style="font-size:1.2rem;color:var(--secondary);"></i>
      <span style="font-size:.85rem;color:#fff;font-weight:600;">
        Welcome, Chef {{ session.get('username', '')|title }}!
      </span>
    </div>
    <a href="/logout" class="btn btn-sm w-100"
       style="background:rgba(255,255,255,.1);color:#fff;border:1px solid rgba(255,255,255,.2);border-radius:7px;font-size:.82rem;">
      <i class="bi bi-box-arrow-right me-1"></i>Logout
    </a>
  </div>
  <div class="sidebar-footer">Powered by Groq + LLaMA 3.3 🚀</div>
</div>

<!-- ══ Main Content ═════════════════════════════════════════════ -->
<div id="main">
  <!-- Mobile topbar toggle -->
  <div class="topbar">
    <button class="btn btn-sm btn-outline-secondary menu-toggle d-none" onclick="toggleSidebar()">
      <i class="bi bi-list" style="font-size:1.2rem;"></i>
    </button>
    <h2 style="flex:1;"><i class="{{ page_icon }}"></i> {{ page_title }}</h2>
    <!-- ── Theme toggle button ─────────────────────────── -->
    <button id="themeToggle" onclick="toggleTheme()" title="Toggle light / dark mode">
      <i id="themeIcon" class="bi bi-moon-stars-fill"></i>
    </button>
  </div>

  {% block content %}{% endblock %}
</div>

<!-- Bootstrap JS -->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
<script>
  // Mobile sidebar toggle
  function toggleSidebar() {
    document.getElementById('sidebar').classList.toggle('open');
  }
  // Show / hide loading overlay
  function showLoading()  { document.getElementById('loadingOverlay').classList.add('show'); }
  function hideLoading()  { document.getElementById('loadingOverlay').classList.remove('show'); }

  // Render Markdown response safely into a target element
  function renderMarkdown(elementId, mdText) {
    const el = document.getElementById(elementId);
    if (!el) return;
    el.innerHTML = marked.parse(mdText);
  }

  // ── Theme toggle ────────────────────────────────────────────────
  const THEME_KEY = 'chefmate_theme';

  function applyTheme(theme) {
    document.documentElement.setAttribute('data-bs-theme', theme);
    localStorage.setItem(THEME_KEY, theme);
    const icon = document.getElementById('themeIcon');
    if (icon) {
      icon.className = theme === 'dark' ? 'bi bi-sun-fill' : 'bi bi-moon-stars-fill';
    }
  }

  function toggleTheme() {
    const current = document.documentElement.getAttribute('data-bs-theme') || 'light';
    applyTheme(current === 'dark' ? 'light' : 'dark');
  }

  // Sync icon on page load (theme already applied before paint by inline script)
  (function initThemeIcon() {
    const saved = localStorage.getItem(THEME_KEY) || 'light';
    const icon  = document.getElementById('themeIcon');
    if (icon) {
      icon.className = saved === 'dark' ? 'bi bi-sun-fill' : 'bi bi-moon-stars-fill';
    }
  })();
</script>
{% block scripts %}{% endblock %}
</body>
</html>
"""

# ══════════════════════════════════════════════════════════════════
#  PAGE TEMPLATES
# ══════════════════════════════════════════════════════════════════

HOME_HTML = BASE_HTML.replace("{% block content %}{% endblock %}", """
<!-- Hero -->
<div class="hero-card">
  <h1>Welcome to ChefMate AI 🍴</h1>
  <p>Your intelligent culinary companion — powered by open-source AI.<br>
     Ask food questions, generate custom recipes, or discover smart ingredient swaps.</p>
  <a href="/chat"   class="btn btn-light fw-bold me-2"><i class="bi bi-chat-dots me-1"></i>Start Cooking Chat</a>
  <a href="/pantry" class="btn btn-outline-light fw-bold"><i class="bi bi-basket3 me-1"></i>Pantry Helper</a>
</div>

<!-- Feature Cards -->
<div class="row g-4">
  <div class="col-md-6">
    <div class="feature-card">
      <div class="feature-icon" style="background:#fff3e0;">💬</div>
      <h5 class="fw-bold">Recipe Chat &amp; Knowledge Agent</h5>
      <p class="text-muted mb-3">Ask anything about cooking — Indian dishes, global cuisines, nutrition, food science, flavour pairings, and practical tips.</p>
      <a href="/chat" class="btn btn-chef btn-sm">Open Chat →</a>
    </div>
  </div>
  <div class="col-md-6">
    <div class="feature-card">
      <div class="feature-icon" style="background:#e8f5e9;">🧺</div>
      <h5 class="fw-bold">Pantry &amp; Substitution Helper</h5>
      <p class="text-muted mb-3">Tell us what's in your fridge, your dietary needs, and how much time you have. We'll generate a complete personalised recipe.</p>
      <a href="/pantry" class="btn btn-chef btn-sm">Open Pantry →</a>
    </div>
  </div>
  <div class="col-md-6">
    <div class="feature-card">
      <div class="feature-icon" style="background:#fce4ec;">📊</div>
      <h5 class="fw-bold">Nutritional Insights</h5>
      <p class="text-muted mb-3">Every generated recipe includes estimated calories, macronutrients (protein, carbs, fats), and a shopping list for missing items.</p>
    </div>
  </div>
  <div class="col-md-6">
    <div class="feature-card">
      <div class="feature-icon" style="background:#e3f2fd;">🔄</div>
      <h5 class="fw-bold">Smart Substitutions</h5>
      <p class="text-muted mb-3">Out of an ingredient? ChefMate suggests the best swap while keeping your dietary preferences and flavour profile intact.</p>
    </div>
  </div>
</div>
""").replace("{% block scripts %}{% endblock %}", "")


CHAT_HTML = BASE_HTML.replace("{% block content %}{% endblock %}", """
<div class="row g-4">
  <!-- Input Panel -->
  <div class="col-lg-5">
    <div class="card h-100">
      <div class="card-header-custom">
        <i class="bi bi-chat-dots me-2"></i>Ask ChefMate AI
      </div>
      <div class="card-body p-4">
        <p class="text-muted small mb-3">Ask cooking questions, get tips on Indian &amp; global cuisines, nutrition advice, and food science explanations.</p>

        <div class="mb-3">
          <label class="form-label fw-semibold">💡 Quick Prompts</label>
          <div id="quickChips">
            <span class="diet-badge" onclick="setPrompt(this)">🫓 Best bread for keto?</span>
            <span class="diet-badge" onclick="setPrompt(this)">🌶️ How to reduce spice in curry?</span>
            <span class="diet-badge" onclick="setPrompt(this)">🥗 High-protein vegetarian meals</span>
            <span class="diet-badge" onclick="setPrompt(this)">🍚 Why does rice go mushy?</span>
            <span class="diet-badge" onclick="setPrompt(this)">🍋 Substitute for lemon juice?</span>
            <span class="diet-badge" onclick="setPrompt(this)">🥦 Nutrition in broccoli</span>
          </div>
        </div>

        <div class="mb-3">
          <label for="chatInput" class="form-label fw-semibold">Your Question</label>
          <textarea id="chatInput" class="form-control" rows="5"
            placeholder="e.g. What are the health benefits of turmeric? How do I make a fluffy omelette?"></textarea>
          <div class="mt-2">
            <button id="voiceBtn" class="btn btn-outline-secondary btn-sm" onclick="toggleVoice()" title="Click to speak your question">
              <i class="bi bi-mic me-1"></i>🎙️ Voice Input
            </button>
            <span id="voiceStatus" class="text-muted small ms-2" style="display:none;"></span>
          </div>
        </div>

        <button class="btn btn-chef w-100" onclick="sendChat()">
          <i class="bi bi-send me-2"></i>Ask ChefMate
        </button>
      </div>
    </div>
  </div>

  <!-- Response Panel -->
  <div class="col-lg-7">
    <div class="card h-100">
      <div class="card-header-custom">
        <i class="bi bi-robot me-2"></i>ChefMate's Answer
      </div>
      <div class="card-body p-4">
        <div id="chatResponse" class="response-box">
          <span class="text-muted"><i class="bi bi-arrow-left-circle me-1"></i>Your answer will appear here…</span>
        </div>
      </div>
    </div>
  </div>
</div>
""").replace("{% block scripts %}{% endblock %}", """
<script>
function setPrompt(el) {
  // Strip emoji + set textarea text
  document.getElementById('chatInput').value = el.textContent.replace(/^[\\p{Emoji}\\s]+/u, '').trim();
  document.querySelectorAll('.diet-badge').forEach(b => b.classList.remove('selected'));
  el.classList.add('selected');
}

async function sendChat() {
  const question = document.getElementById('chatInput').value.trim();
  if (!question) { alert('Please enter a question.'); return; }
  showLoading();
  try {
    const res  = await fetch('/api/chat', {
      method:  'POST',
      headers: {'Content-Type': 'application/json'},
      body:    JSON.stringify({ message: question })
    });
    const data = await res.json();
    renderMarkdown('chatResponse', data.response || data.error || 'No response received.');
  } catch (err) {
    document.getElementById('chatResponse').innerText = '⚠️ Network error: ' + err.message;
  } finally {
    hideLoading();
  }
}

// Allow Ctrl+Enter to submit
document.getElementById('chatInput').addEventListener('keydown', function(e) {
  if (e.ctrlKey && e.key === 'Enter') sendChat();
});

// ── Voice Input ──────────────────────────────────────────────────────────────
(function() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const btn    = document.getElementById('voiceBtn');
  const status = document.getElementById('voiceStatus');

  if (!SpeechRecognition) {
    btn.disabled = true;
    btn.title = 'Voice input is not supported in this browser.';
    btn.innerHTML = '<i class="bi bi-mic-mute me-1"></i>Voice Input (unsupported)';
    return;
  }

  const recognition = new SpeechRecognition();
  recognition.lang = 'en-US';
  recognition.interimResults = true;
  recognition.maxAlternatives = 1;
  let listening = false;

  recognition.onstart = function() {
    listening = true;
    btn.classList.replace('btn-outline-secondary', 'btn-danger');
    btn.innerHTML = '<i class="bi bi-stop-circle me-1"></i>Stop Recording';
    status.style.display = 'inline';
    status.textContent = '🔴 Listening…';
  };

  recognition.onresult = function(e) {
    let transcript = '';
    for (let i = e.resultIndex; i < e.results.length; i++) {
      transcript += e.results[i][0].transcript;
    }
    document.getElementById('chatInput').value = transcript;
  };

  recognition.onerror = function(e) {
    status.textContent = '⚠️ ' + (e.error === 'not-allowed' ? 'Microphone access denied.' : e.error);
  };

  recognition.onend = function() {
    listening = false;
    btn.classList.replace('btn-danger', 'btn-outline-secondary');
    btn.innerHTML = '<i class="bi bi-mic me-1"></i>🎙️ Voice Input';
    status.style.display = 'none';
  };

  window.toggleVoice = function() {
    if (listening) { recognition.stop(); } else { recognition.start(); }
  };
})();
</script>
""")


PANTRY_HTML = BASE_HTML.replace("{% block content %}{% endblock %}", """
<div class="row g-4">
  <!-- Input Panel -->
  <div class="col-lg-5">
    <div class="card">
      <div class="card-header-custom">
        <i class="bi bi-basket3 me-2"></i>Pantry & Preferences
      </div>
      <div class="card-body p-4">

        <!-- Ingredients -->
        <div class="mb-3">
          <label for="ingredients" class="form-label fw-semibold">🥕 Ingredients You Have</label>
          <textarea id="ingredients" class="form-control" rows="3"
            placeholder="e.g. chicken breast, garlic, olive oil, tomatoes, onion, spinach, cumin…"></textarea>
          <div class="form-text">Separate with commas. Be as specific as you like!</div>
        </div>

        <!-- Dietary Preferences -->
        <div class="mb-3">
          <label class="form-label fw-semibold">🌿 Dietary Preferences</label>
          <div id="dietChips">
            <span class="diet-badge" onclick="toggleDiet(this)">🌱 Vegan</span>
            <span class="diet-badge" onclick="toggleDiet(this)">🥛 Vegetarian</span>
            <span class="diet-badge" onclick="toggleDiet(this)">🥩 Non-Vegetarian</span>
            <span class="diet-badge" onclick="toggleDiet(this)">🌾 Gluten-Free</span>
            <span class="diet-badge" onclick="toggleDiet(this)">🥛 Dairy-Free</span>
            <span class="diet-badge" onclick="toggleDiet(this)">⚡ Low-Carb / Keto</span>
            <span class="diet-badge" onclick="toggleDiet(this)">🚫 Sugar-Free</span>
            <span class="diet-badge" onclick="toggleDiet(this)">🫀 Low-Sodium</span>
          </div>
          <input type="text" id="dietOther" class="form-control mt-2"
                 placeholder="Or type custom preference (e.g. nut-free, Jain)"/>
        </div>

        <!-- Cuisine -->
        <div class="mb-3">
          <label for="cuisine" class="form-label fw-semibold">🌍 Preferred Cuisine (optional)</label>
          <select id="cuisine" class="form-select">
            <option value="">Any / Surprise me!</option>
            <option>Indian</option>
            <option>South Indian</option>
            <option>North Indian</option>
            <option>Mediterranean</option>
            <option>Chinese</option>
            <option>Italian</option>
            <option>Mexican</option>
            <option>Thai</option>
            <option>Middle Eastern</option>
            <option>American</option>
          </select>
        </div>

        <!-- Time -->
        <div class="mb-3">
          <label for="cookTime" class="form-label fw-semibold">⏱️ Max Cooking Time</label>
          <select id="cookTime" class="form-select">
            <option value="15 minutes">⚡ 15 minutes (Quick)</option>
            <option value="30 minutes" selected>🕐 30 minutes</option>
            <option value="45 minutes">🕞 45 minutes</option>
            <option value="1 hour">⏰ 1 hour</option>
            <option value="No time limit">🍲 No limit (slow-cook OK)</option>
          </select>
        </div>

        <!-- Servings -->
        <div class="mb-4">
          <label for="servings" class="form-label fw-semibold">👥 Servings</label>
          <input type="number" id="servings" class="form-control" value="2" min="1" max="20"/>
        </div>

        <button class="btn btn-chef w-100" onclick="generateRecipe()">
          <i class="bi bi-magic me-2"></i>Generate My Recipe
        </button>
      </div>
    </div>

    <!-- ── Past Recipes History ───────────────────────────────── -->
    <div class="card mt-4" id="historyCard">
      <div class="card-header-custom d-flex align-items-center justify-content-between">
        <span><i class="bi bi-clock-history me-2"></i>📜 Past Recipes</span>
        <button class="btn btn-sm btn-outline-light" onclick="clearHistory()" title="Clear all history">
          <i class="bi bi-trash me-1"></i>Clear
        </button>
      </div>
      <div class="card-body p-0">
        <!-- Empty state -->
        <div id="historyEmpty" class="text-center text-muted py-4 px-3" style="font-size:.88rem;">
          <i class="bi bi-journal-x" style="font-size:1.6rem;display:block;margin-bottom:6px;"></i>
          No recipes saved yet.<br/>Generate one above to see it here!
        </div>
        <!-- History list – populated by JS -->
        <ul class="list-group list-group-flush" id="historyList" style="display:none;"></ul>
      </div>
    </div>
  </div>

  <!-- Recipe Output Panel -->
  <div class="col-lg-7">
    <div class="card">
      <div class="card-header-custom">
        <i class="bi bi-journal-richtext me-2"></i>Your Personalised Recipe
        <!-- Badge shows which history slot is being viewed -->
        <span id="historyViewingBadge" class="badge ms-2 d-none"
              style="background:rgba(255,255,255,.25);font-size:.75rem;font-weight:500;">
          <i class="bi bi-clock-history me-1"></i><span id="historyViewingLabel"></span>
        </span>
      </div>
      <div class="card-body p-4">
        <div id="recipeResponse" class="response-box">
          <span class="text-muted"><i class="bi bi-arrow-left-circle me-1"></i>Fill in your pantry details and click <strong>Generate My Recipe</strong>…</span>
        </div>

        <!-- Action buttons (shown after recipe generates) -->
        <div id="recipeActions" class="mt-3 d-none">
          <button class="btn btn-sm btn-outline-secondary me-2" onclick="copyRecipe()">
            <i class="bi bi-clipboard me-1"></i>Copy Recipe
          </button>
          <button class="btn btn-sm btn-outline-danger" onclick="clearRecipe()">
            <i class="bi bi-trash me-1"></i>Clear
          </button>
        </div>
      </div>
    </div>
  </div>
</div>
""").replace("{% block scripts %}{% endblock %}", """
<script>
// ── Constants ──────────────────────────────────────────────────
const HISTORY_KEY  = 'chefmate_recipe_history';
const HISTORY_MAX  = 10;   // keep last N recipes

// ── State ──────────────────────────────────────────────────────
let selectedDiets  = [];
let lastRecipeText = '';

// ── Diet badge toggle ──────────────────────────────────────────
function toggleDiet(el) {
  el.classList.toggle('selected');
  const label = el.textContent.trim().replace(/^[^\\w]+/, '').trim();
  if (el.classList.contains('selected')) {
    if (!selectedDiets.includes(label)) selectedDiets.push(label);
  } else {
    selectedDiets = selectedDiets.filter(d => d !== label);
  }
}

// ── localStorage helpers ───────────────────────────────────────
function loadHistory() {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]'); }
  catch { return []; }
}

function saveHistory(arr) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(arr));
}

/**
 * Extract the recipe title from the Markdown text.
 * Looks for the first "### 🍳 ..." heading; falls back to first 60 chars.
 */
function extractTitle(mdText) {
  const m = mdText.match(/###\\s*[^\\n]*?Adapted Recipe Title[^\\n]*\\n([^\\n]+)/i)
         || mdText.match(/###\\s*🍳\\s*([^\\n]+)/i)
         || mdText.match(/^###\\s*(.+)/m);
  if (m) return m[1].trim().replace(/\\*+/g, '').trim();
  return mdText.slice(0, 60).replace(/\\n/g, ' ').trim() + '…';
}

/**
 * Push a new recipe into history (prepend; cap at HISTORY_MAX).
 */
function pushHistory(ingredients, mdText) {
  const history = loadHistory();
  const entry = {
    title:       extractTitle(mdText),
    ingredients: ingredients,
    timestamp:   new Date().toLocaleString(),
    content:     mdText,
  };
  history.unshift(entry);          // newest first
  if (history.length > HISTORY_MAX) history.length = HISTORY_MAX;
  saveHistory(history);
  renderHistoryList();
}

/**
 * Render the history list UI from localStorage.
 */
function renderHistoryList() {
  const history = loadHistory();
  const list    = document.getElementById('historyList');
  const empty   = document.getElementById('historyEmpty');

  list.innerHTML = '';

  if (!history.length) {
    empty.style.display = '';
    list.style.display  = 'none';
    return;
  }

  empty.style.display = 'none';
  list.style.display  = '';

  history.forEach((item, idx) => {
    const li = document.createElement('li');
    li.className = 'list-group-item list-group-item-action px-3 py-2';
    li.style.cssText = 'cursor:pointer;border-left:3px solid transparent;transition:border-color .15s,background .15s;';
    li.innerHTML = `
      <div class="d-flex align-items-start gap-2">
        <span style="font-size:1.1rem;line-height:1.4;">🍽️</span>
        <div style="min-width:0;">
          <div class="fw-semibold text-truncate" style="font-size:.88rem;max-width:190px;"
               title="${item.title.replace(/"/g, '&quot;')}">${item.title}</div>
          <div class="text-muted" style="font-size:.75rem;">
            <i class="bi bi-clock me-1"></i>${item.timestamp}
          </div>
          <div class="text-muted text-truncate" style="font-size:.75rem;max-width:190px;"
               title="${item.ingredients.replace(/"/g, '&quot;')}">
            <i class="bi bi-basket me-1"></i>${item.ingredients}
          </div>
        </div>
      </div>`;

    // Hover highlight
    li.addEventListener('mouseenter', () => {
      li.style.background   = 'rgba(232,93,4,.06)';
      li.style.borderColor  = 'var(--primary)';
    });
    li.addEventListener('mouseleave', () => {
      li.style.background  = '';
      li.style.borderColor = 'transparent';
    });

    // Click → load recipe into result panel
    li.addEventListener('click', () => loadFromHistory(idx));

    list.appendChild(li);
  });
}

/**
 * Load a history entry into the recipe result panel.
 */
function loadFromHistory(idx) {
  const history = loadHistory();
  const item    = history[idx];
  if (!item) return;

  lastRecipeText = item.content;
  renderMarkdown('recipeResponse', lastRecipeText);
  document.getElementById('recipeActions').classList.remove('d-none');

  // Show the "viewing from history" badge
  const badge = document.getElementById('historyViewingBadge');
  document.getElementById('historyViewingLabel').textContent = item.timestamp;
  badge.classList.remove('d-none');

  // Scroll result panel into view on mobile
  document.getElementById('recipeResponse').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/**
 * Clear all saved history.
 */
function clearHistory() {
  if (!confirm('Clear all saved recipe history? This cannot be undone.')) return;
  localStorage.removeItem(HISTORY_KEY);
  renderHistoryList();
  // If currently viewing a history item, reset the badge
  document.getElementById('historyViewingBadge').classList.add('d-none');
}

// ── Main recipe generation ─────────────────────────────────────
async function generateRecipe() {
  const ingredients = document.getElementById('ingredients').value.trim();
  if (!ingredients) { alert('Please enter at least one ingredient.'); return; }

  const dietOther  = document.getElementById('dietOther').value.trim();
  const allDiets   = [...selectedDiets, ...(dietOther ? [dietOther] : [])];
  const cuisine    = document.getElementById('cuisine').value;
  const cookTime   = document.getElementById('cookTime').value;
  const servings   = document.getElementById('servings').value;

  const message = [
    `Ingredients I have: ${ingredients}`,
    allDiets.length  ? `Dietary preferences: ${allDiets.join(', ')}` : '',
    cuisine          ? `Preferred cuisine: ${cuisine}`               : '',
    `Max cooking time: ${cookTime}`,
    `Servings: ${servings}`,
  ].filter(Boolean).join('\\n');

  // Hide any "viewing history" badge for fresh generation
  document.getElementById('historyViewingBadge').classList.add('d-none');

  showLoading();
  try {
    const res  = await fetch('/api/recipe', {
      method:  'POST',
      headers: {'Content-Type': 'application/json'},
      body:    JSON.stringify({ message })
    });
    const data = await res.json();
    lastRecipeText = data.response || data.error || 'No response received.';
    renderMarkdown('recipeResponse', lastRecipeText);
    document.getElementById('recipeActions').classList.remove('d-none');

    // ── Save to history ──────────────────────────────────────
    if (data.response) {
      pushHistory(ingredients, lastRecipeText);
    }
  } catch (err) {
    document.getElementById('recipeResponse').innerText = '⚠️ Network error: ' + err.message;
  } finally {
    hideLoading();
  }
}

// ── Existing helpers (unchanged) ──────────────────────────────
function copyRecipe() {
  navigator.clipboard.writeText(lastRecipeText)
    .then(() => alert('Recipe copied to clipboard! 📋'))
    .catch(() => alert('Could not copy. Please select and copy manually.'));
}

function clearRecipe() {
  document.getElementById('recipeResponse').innerHTML =
    '<span class="text-muted">Fill in your pantry details and click <strong>Generate My Recipe</strong>…</span>';
  document.getElementById('recipeActions').classList.add('d-none');
  document.getElementById('historyViewingBadge').classList.add('d-none');
  lastRecipeText = '';
}

// ── Boot: render history on page load ─────────────────────────
document.addEventListener('DOMContentLoaded', renderHistoryList);
</script>
""")


ABOUT_HTML = BASE_HTML.replace("{% block content %}{% endblock %}", """
<div class="row g-4">
  <div class="col-lg-8">
    <div class="card mb-4">
      <div class="card-header-custom"><i class="bi bi-info-circle me-2"></i>About ChefMate AI</div>
      <div class="card-body p-4">
        <p>
          <strong>ChefMate AI</strong> is an open-source, hackathon-ready intelligent recipe generator
          built with <strong>Python Flask</strong>, <strong>Groq API</strong> (llama-3.3-70b-versatile), and
          <strong>Bootstrap 5</strong>. It runs entirely on free, open-source LLM models — no paid APIs needed.
        </p>
        <hr/>
        <h6 class="fw-bold text-danger mb-3">🧠 AI Agents</h6>
        <div class="row g-3 mb-4">
          <div class="col-md-6">
            <div class="p-3 rounded-3" style="background:#fff3e0;border:1px solid #ffe0b2;">
              <strong>💬 Knowledge Agent</strong><br/>
              <small class="text-muted">Answers cooking questions, explains nutrition, suggests pairings, and gives practical tips across Indian &amp; global cuisines.</small>
            </div>
          </div>
          <div class="col-md-6">
            <div class="p-3 rounded-3" style="background:#e8f5e9;border:1px solid #c8e6c9;">
              <strong>🧺 Recipe &amp; Substitution Agent</strong><br/>
              <small class="text-muted">Generates structured, personalised recipes from your available ingredients, dietary needs, and time constraints.</small>
            </div>
          </div>
        </div>

        <h6 class="fw-bold text-danger mb-3">🛠️ Tech Stack</h6>
        <div class="table-responsive">
          <table class="table table-sm table-bordered">
            <thead class="table-light"><tr><th>Layer</th><th>Technology</th></tr></thead>
            <tbody>
              <tr><td>Backend</td><td>Python 3.10+, Flask 3.x</td></tr>
              <tr><td>AI / LLM</td><td>Groq API — llama-3.3-70b-versatile (free tier)</td></tr>
              <tr><td>Frontend</td><td>Bootstrap 5.3, Bootstrap Icons, Marked.js</td></tr>
              <tr><td>Env Management</td><td>python-dotenv</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-header-custom"><i class="bi bi-rocket me-2"></i>Quick Setup Guide</div>
      <div class="card-body p-4">
        <ol class="mb-0" style="line-height:2;">
          <li>Clone or download this project.</li>
          <li>Install dependencies: <code>pip install -r requirements.txt</code></li>
          <li>Copy <code>.env.example</code> → <code>.env</code> and add your <strong>Groq API key</strong>.</li>
          <li>Run: <code>python app.py</code></li>
          <li>Open <a href="http://127.0.0.1:5000" target="_blank">http://127.0.0.1:5000</a> 🎉</li>
        </ol>
        <div class="mt-3 p-3 rounded-3" style="background:#e8f5e9;border:1px solid #a5d6a7;">
          <strong>🔑 Get a free Groq API key:</strong><br/>
          Visit <a href="https://console.groq.com" target="_blank">console.groq.com</a>, sign up for free,
          and create an API key. The LLaMA 3 model is available at no cost on the free tier.
        </div>
      </div>
    </div>
  </div>

  <div class="col-lg-4">
    <div class="card mb-4">
      <div class="card-header-custom"><i class="bi bi-lightbulb me-2"></i>Sample Questions to Try</div>
      <div class="card-body p-3">
        <div class="tip-chip">💡 What makes dal makhani creamy?</div>
        <div class="tip-chip">💡 Best substitute for eggs in baking?</div>
        <div class="tip-chip">💡 How to make crispy dosa batter?</div>
        <div class="tip-chip">💡 Low-carb alternative to rice?</div>
        <div class="tip-chip">💡 Why does bread need to rise twice?</div>
        <div class="tip-chip">💡 Protein content of paneer vs tofu?</div>
      </div>
    </div>

    <div class="card">
      <div class="card-header-custom"><i class="bi bi-star me-2"></i>Features</div>
      <div class="card-body p-3">
        <ul class="list-unstyled mb-0" style="line-height:2.2;">
          <li><i class="bi bi-check2-circle text-success me-2"></i>Structured recipe output</li>
          <li><i class="bi bi-check2-circle text-success me-2"></i>Smart ingredient substitutions</li>
          <li><i class="bi bi-check2-circle text-success me-2"></i>Nutritional estimates</li>
          <li><i class="bi bi-check2-circle text-success me-2"></i>Shopping list generation</li>
          <li><i class="bi bi-check2-circle text-success me-2"></i>Dietary preference filters</li>
          <li><i class="bi bi-check2-circle text-success me-2"></i>Multi-cuisine support</li>
          <li><i class="bi bi-check2-circle text-success me-2"></i>Copy recipe to clipboard</li>
          <li><i class="bi bi-check2-circle text-success me-2"></i>Mobile responsive UI</li>
        </ul>
      </div>
    </div>
  </div>
</div>
""").replace("{% block scripts %}{% endblock %}", "")


# ══════════════════════════════════════════════════════════════════
#  ROUTE HELPERS  – inject active page + icon into base template
# ══════════════════════════════════════════════════════════════════

def render_page(template_str, active, page_title, page_icon):
    """Render a page with the correct sidebar highlight and title."""
    return render_template_string(
        template_str,
        active=active,
        page_title=page_title,
        page_icon=page_icon,
        session=session,   # pass session so Jinja can read username
    )


# ── Login-required decorator ──────────────────────────────────────
def login_required(f):
    """Redirect to /login if the user is not logged in."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if "username" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# ══════════════════════════════════════════════════════════════════
#  FLASK ROUTES  – Auth (Login / Logout)
# ══════════════════════════════════════════════════════════════════

@app.route("/login", methods=["GET", "POST"])
def login():
    """Show the login form (GET) or validate credentials (POST)."""
    # Already logged in → go home
    if "username" in session:
        return redirect(url_for("home"))

    error    = None
    success  = request.args.get("success")   # set after successful registration
    username = ""

    if request.method == "POST":
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "")

        if username in users_db and users_db[username] == password:
            session["username"] = username
            return redirect(url_for("home"))
        else:
            error = "Invalid username or password. Please try again."

    return render_template_string(LOGIN_HTML, error=error, success=success, username=username)


@app.route("/register", methods=["GET", "POST"])
def register():
    """Show the registration form (GET) or create a new account (POST)."""
    # Already logged in → go home
    if "username" in session:
        return redirect(url_for("home"))

    error    = None
    username = ""

    if request.method == "POST":
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm",  "")

        # ── Server-side validations ────────────────────────────────
        if len(username) < 3:
            error = "Username must be at least 3 characters long."
        elif not username.replace("_", "").isalnum():
            error = "Username may only contain letters, numbers, and underscores."
        elif len(password) < 4:
            error = "Password must be at least 4 characters long."
        elif password != confirm:
            error = "Passwords do not match. Please try again."
        elif username in users_db:
            error = f"Username '{username}' is already taken. Please choose another."
        else:
            # ── All checks passed: register the user ──────────────
            users_db[username] = password
            return redirect(url_for("login") + "?success=Account+created+successfully!+You+can+now+sign+in.")

    return render_template_string(REGISTER_HTML, error=error, username=username)


@app.route("/logout")
def logout():
    """Clear the session and redirect to login."""
    session.clear()
    return redirect(url_for("login"))


# ══════════════════════════════════════════════════════════════════
#  FLASK ROUTES  – Pages  (all protected by @login_required)
# ══════════════════════════════════════════════════════════════════

@app.route("/")
@login_required
def home():
    return render_page(HOME_HTML, "home", "Home", "bi bi-house-fill")


@app.route("/chat")
@login_required
def chat():
    return render_page(CHAT_HTML, "chat", "Recipe Chat", "bi bi-chat-dots-fill")


@app.route("/pantry")
@login_required
def pantry():
    return render_page(PANTRY_HTML, "pantry", "Pantry Helper", "bi bi-basket3-fill")


@app.route("/about")
@login_required
def about():
    return render_page(ABOUT_HTML, "about", "About", "bi bi-info-circle-fill")


# ══════════════════════════════════════════════════════════════════
#  FLASK ROUTES  – API Endpoints
# ══════════════════════════════════════════════════════════════════

@app.route("/api/chat", methods=["POST"])
@login_required
def api_chat():
    """
    Agent 1 – Nutrition & Recipe Knowledge Agent.
    Accepts: { "message": "..." }
    Returns: { "response": "..." }
    """
    data    = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "⚠️ No message provided."}), 400

    response = generate_response(
        system_prompt=KNOWLEDGE_AGENT_PROMPT,
        user_message=message,
        max_tokens=4096,
    )
    return jsonify({"response": response})


@app.route("/api/recipe", methods=["POST"])
@login_required
def api_recipe():
    """
    Agent 2 – Personalised Recipe & Substitution Agent.
    Accepts: { "message": "Ingredients I have: ...\nDietary: ..." }
    Returns: { "response": "..." }
    """
    data    = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "⚠️ No message provided."}), 400

    response = generate_response(
        system_prompt=RECIPE_AGENT_PROMPT,
        user_message=message,
        max_tokens=4096,   # recipes need more tokens
    )
    return jsonify({"response": response})


# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("  🍽️  ChefMate AI – Intelligent Recipe Generator")
    print("=" * 55)
    if not GROQ_API_KEY:
        print("  ⚠️  WARNING: GROQ_API_KEY not set in .env file!")
        print("  → Copy .env.example to .env and add your key.")
    else:
        print(f"  ✅ Groq API key loaded  |  Model: {GROQ_MODEL}")
    print("  🚀 Starting server at http://127.0.0.1:5000")
    print("=" * 55)
    app.run(debug=True, host="127.0.0.1", port=5000)
