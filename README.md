# Fingerprint-Based Blood Group Prediction

Split into two independently deployable pieces:

```
backend/    Flask API  → deploy on Render
frontend/   Static site → deploy on Vercel
```

⚠️ **Disclaimer:** experimental ML demo, not a medical diagnostic tool.

---

## 1. Add your model files

Copy your trained `.pkl` files into `backend/model/`:
```
backend/model/voting_classifier.pkl
backend/model/scaler.pkl
backend/model/label_encoder.pkl
```
See `backend/model/README.md` for details (including a note on
scikit-learn version matching, and Git LFS if the files are large).

## 2. Push to GitHub

```bash
cd project
git init
git add .
git commit -m "Initial commit: split backend/frontend"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

## 3. Deploy the backend on Render

1. Go to [render.com](https://render.com) → **New +** → **Web Service**
2. Connect your GitHub repo
3. Set **Root Directory** to `backend`
4. Render should auto-detect `render.yaml`, or set manually:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT`
   - **Environment:** Python 3.9
5. Deploy. Once live, note your backend URL, e.g.
   `https://fingerprint-blood-prediction-api.onrender.com`
6. Test it: visit `<your-backend-url>/health` — should return
   `{"status": "ok", "model_loaded": true}`

## 4. Point the frontend at the backend

Edit `frontend/config.js`:
```js
const API_BASE_URL = "https://fingerprint-blood-prediction-api.onrender.com";
```
Commit and push this change.

## 5. Deploy the frontend on Vercel

1. Go to [vercel.com](https://vercel.com) → **Add New** → **Project**
2. Import the same GitHub repo
3. Set **Root Directory** to `frontend`
4. Framework preset: **Other** (it's plain HTML/CSS/JS, no build step)
5. Deploy. Vercel will give you a URL like `https://your-app.vercel.app`

## 6. Lock down CORS (recommended)

Back in Render, set the `FRONTEND_ORIGIN` environment variable to your
actual Vercel URL instead of `*`:
```
FRONTEND_ORIGIN=https://your-app.vercel.app
```
Redeploy the backend for this to take effect. This restricts the API to
only accept requests from your frontend, instead of any website.

## 7. Test end to end

Open your Vercel URL, upload a fingerprint image, and confirm a
prediction comes back. If you get a CORS error in the browser console,
double check `FRONTEND_ORIGIN` matches your Vercel URL exactly
(including `https://`, no trailing slash).

---

## Local development

**Backend:**
```bash
cd backend
pip install -r requirements.txt
python app.py
# runs on http://localhost:5000
```

**Frontend:**
Just open `frontend/index.html` directly in a browser, or serve it:
```bash
cd frontend
python -m http.server 8000
# visit http://localhost:8000
```
Keep `config.js` pointed at `http://localhost:5000` for local testing.
