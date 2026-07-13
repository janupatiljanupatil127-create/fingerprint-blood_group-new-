# Model artifacts go here

This folder needs three files before `app.py` will work:

```
model/
├── voting_classifier.pkl   # the trained VotingClassifier
├── scaler.pkl               # the fitted StandardScaler
└── label_encoder.pkl        # the fitted LabelEncoder (blood group <-> class index)
```

Copy your existing `.pkl` files from the Hugging Face Space's `model/`
(or wherever your training script saved them) into this folder before
pushing to GitHub / deploying to Render.

**Important:** these files were pickled with **scikit-learn 1.7.2**
(per the version warnings seen on the original deployment). The
`requirements.txt` in this repo pins `scikit-learn==1.7.2` to match —
don't change that version unless you also re-train and re-save these
`.pkl` files with the new version, or you'll get `InconsistentVersionWarning`
again (and possibly broken predictions).

**Git note:** `.pkl` files can get large. If any of these exceed ~50MB,
use Git LFS:
```bash
git lfs track "*.pkl"
git add .gitattributes
```
