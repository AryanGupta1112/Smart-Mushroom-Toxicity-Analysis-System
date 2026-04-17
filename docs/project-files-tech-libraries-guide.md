# Project Files, Tech Stack, and Library Purpose Guide

This guide explains the purpose and goal of each major file, technology, and library used in this project in very simple English.

## 1) Tech Stack (Simple Purpose)

| Tech | Purpose (1-2 lines) |
|---|---|
| Python | Main backend language used for APIs, ML training, and inference. |
| FastAPI | Creates backend REST APIs quickly with clean request/response handling. |
| Pydantic | Validates API input/output data so wrong values are rejected safely. |
| scikit-learn | Trains and runs classical ML models (LR, DT, RF, KNN, SVM). |
| SHAP | Explains which input features influenced a prediction the most. |
| SQLite | Stores prediction history in a lightweight local database. |
| SQLAlchemy | Handles database table models and queries in Python code. |
| React + TypeScript | Builds the frontend UI with typed, maintainable components. |
| Vite | Fast frontend dev/build tool for React projects. |
| Tailwind CSS | Utility-first styling system for clean UI design. |
| shadcn/ui + Radix UI | Ready, accessible UI components (cards, dialogs, selects, etc.). |
| Recharts | Draws charts for model scores, trends, and factor visuals. |
| TanStack Query | Manages API fetching, caching, and refresh behavior on frontend. |
| Docker Compose | Runs backend/front-end containers consistently (when needed). |

---

## 2) Backend Libraries (`backend/requirements.txt`)

| Library | Purpose (1-2 lines) |
|---|---|
| `fastapi` | Backend API framework for endpoints like `/api/predict` and `/api/history`. |
| `uvicorn[standard]` | ASGI server used to run FastAPI app locally/in containers. |
| `pydantic` | Data models and validation for request and response schemas. |
| `pydantic-settings` | Reads configuration from `.env` into typed settings class. |
| `scikit-learn` | ML pipelines, preprocessing, training, evaluation, and inference. |
| `pandas` | Tabular data loading and transformations for dataset and inference rows. |
| `numpy` | Numeric operations and arrays used during metrics/explanations. |
| `shap` | Feature contribution explanations for tree-based models. |
| `joblib` | Saves and loads trained model pipelines as `.joblib` files. |
| `sqlalchemy` | ORM and DB engine/session management for SQLite history. |
| `python-multipart` | Supports multipart form handling (useful for API compatibility). |
| `httpx` | HTTP client utility, often used for testing/API interactions. |

---

## 3) Frontend Runtime Libraries (`frontend/package.json` -> `dependencies`)

| Library | Purpose (1-2 lines) |
|---|---|
| `react` | Core UI library for building frontend components. |
| `react-dom` | Renders React app into browser DOM. |
| `react-router-dom` | Handles page routing (dashboard, models, history, etc.). |
| `typescript` (via toolchain) | Adds types to reduce frontend bugs and improve readability. |
| `axios` | Sends HTTP requests from frontend to backend APIs. |
| `@tanstack/react-query` | Fetches/caches API data and supports auto-refresh. |
| `react-hook-form` | Efficient form state handling for long input forms. |
| `@hookform/resolvers` | Connects React Hook Form with Zod validation. |
| `zod` | Frontend schema validation for form values. |
| `recharts` | Chart components for trends and model comparisons. |
| `framer-motion` | Smooth animations and transitions in UI. |
| `lucide-react` | Icon set used in headers/cards/buttons. |
| `next-themes` | Light/dark mode theme switching in frontend. |
| `@radix-ui/react-dialog` | Accessible dialog component base. |
| `@radix-ui/react-label` | Accessible label component base. |
| `@radix-ui/react-select` | Accessible select/dropdown base. |
| `@radix-ui/react-slot` | Slot utility used by shadcn component patterns. |
| `@radix-ui/react-tooltip` | Accessible tooltip base. |
| `class-variance-authority` | Variant-based class style patterns for UI components. |
| `clsx` | Combines conditional class names cleanly. |
| `tailwind-merge` | Merges Tailwind classes safely without conflicts. |
| `@fontsource/manrope` | Imports Manrope font locally for typography. |
| `@fontsource/space-grotesk` | Imports Space Grotesk font locally for typography. |

---

## 4) Frontend Dev Libraries (`frontend/package.json` -> `devDependencies`)

| Library | Purpose (1-2 lines) |
|---|---|
| `vite` | Fast dev server and build tool. |
| `@vitejs/plugin-react` | Enables React support in Vite. |
| `eslint` | Linting rules for code quality and consistency. |
| `@eslint/js` | Base ESLint rule set. |
| `typescript-eslint` | ESLint parser/rules for TypeScript files. |
| `eslint-plugin-react-hooks` | Validates correct React Hooks usage. |
| `eslint-plugin-react-refresh` | Ensures safe Fast Refresh patterns. |
| `@types/node` | Type definitions for Node APIs in TS code. |
| `@types/react` | Type definitions for React APIs. |
| `@types/react-dom` | Type definitions for ReactDOM APIs. |
| `tailwindcss` | Utility CSS framework generator. |
| `postcss` | CSS transformation pipeline tool. |
| `autoprefixer` | Adds vendor prefixes for browser compatibility. |
| `globals` | Standard global variable definitions for linting environments. |

---

## 5) File-by-File Purpose (Project Files)

### Root files

| File | Purpose (1-2 lines) |
|---|---|
| `.gitignore` | Tells Git which files/folders should not be tracked. |
| `README.md` | Main project overview, setup steps, architecture summary. |
| `docker-compose.yml` | Defines how backend/frontend containers run together. |
| `.vscode/settings.json` | Workspace editor settings for local development convenience. |

### Backend top-level files

| File | Purpose (1-2 lines) |
|---|---|
| `backend/.env` | Local backend runtime configuration values. |
| `backend/.env.example` | Sample environment variables template for backend. |
| `backend/Dockerfile` | Container build instructions for backend service. |
| `backend/README.md` | Backend-specific run, train, and API notes. |
| `backend/requirements.txt` | Python dependency list for backend environment. |

### Backend app entry and package files

| File | Purpose (1-2 lines) |
|---|---|
| `backend/app/__init__.py` | Marks `app` as a Python package. |
| `backend/app/main.py` | FastAPI app startup, middleware, router mounting, model readiness. |

### Backend API layer

| File | Purpose (1-2 lines) |
|---|---|
| `backend/app/api/__init__.py` | Marks API folder as package. |
| `backend/app/api/routes.py` | Defines API endpoints (`predict`, `history`, `features`, etc.). |

### Backend core configuration/database

| File | Purpose (1-2 lines) |
|---|---|
| `backend/app/core/__init__.py` | Marks core folder as package. |
| `backend/app/core/config.py` | Loads app settings (paths, model names, flags, CORS). |
| `backend/app/core/database.py` | Creates DB engine/session and base model class. |

### Backend ML files

| File | Purpose (1-2 lines) |
|---|---|
| `backend/app/ml/__init__.py` | Marks ML folder as package. |
| `backend/app/ml/feature_catalog.py` | Canonical feature definitions and allowed value labels. |
| `backend/app/ml/train.py` | Full training pipeline: load data, preprocess, train, evaluate, save artifacts. |

### Backend model/schema/service/util files

| File | Purpose (1-2 lines) |
|---|---|
| `backend/app/models/__init__.py` | Marks models folder as package. |
| `backend/app/models/prediction_history.py` | SQLAlchemy table model for prediction history records. |
| `backend/app/schemas/__init__.py` | Marks schemas folder as package. |
| `backend/app/schemas/risk.py` | Pydantic request/response models and validators. |
| `backend/app/services/__init__.py` | Marks services folder as package. |
| `backend/app/services/model_service.py` | Loads models and performs prediction/explainability/warnings. |
| `backend/app/services/recommendation_service.py` | Rule-based recommendation messages from inputs and probability. |
| `backend/app/services/history_service.py` | Save/list prediction history from SQLite. |
| `backend/app/utils/__init__.py` | Marks utils folder as package. |
| `backend/app/utils/risk.py` | Converts probability to score and score to risk level. |

### Backend data/notebooks/saved artifacts

| File | Purpose (1-2 lines) |
|---|---|
| `backend/data/README.md` | Explains what is stored in backend data folder. |
| `backend/data/prediction_history.db` | SQLite database file containing saved prediction history. |
| `backend/notebooks/README.md` | Notes about notebook usage/location. |
| `backend/saved_models/README.md` | Explains saved model artifact folder content. |
| `backend/saved_models/logistic_regression.joblib` | Trained Logistic Regression pipeline artifact. |
| `backend/saved_models/decision_tree.joblib` | Trained Decision Tree pipeline artifact. |
| `backend/saved_models/random_forest.joblib` | Trained Random Forest pipeline artifact. |
| `backend/saved_models/knn.joblib` | Trained KNN pipeline artifact. |
| `backend/saved_models/svm.joblib` | Trained SVM pipeline artifact. |
| `backend/saved_models/metadata.json` | Training metadata (metrics, features, frequencies, best model). |

### Backend auto-generated cache files

| File group | Purpose (1-2 lines) |
|---|---|
| `backend/app/**/__pycache__/*.pyc` | Auto-generated Python bytecode cache for faster imports. |

### Documentation files

| File | Purpose (1-2 lines) |
|---|---|
| `docs/api-reference.md` | API endpoint summary and request/response documentation. |
| `docs/architecture.md` | High-level system architecture documentation. |
| `docs/project-story.md` | Narrative explanation of complete project journey. |
| `docs/ML_Project_Report.docx` | Full formatted project report document. |

### Frontend top-level files

| File | Purpose (1-2 lines) |
|---|---|
| `frontend/.env` | Frontend environment variables for local setup. |
| `frontend/.env.local` | Local override for frontend API URL and settings. |
| `frontend/Dockerfile` | Container build instructions for frontend service. |
| `frontend/README.md` | Frontend-specific setup/run guide. |
| `frontend/package.json` | Frontend scripts and dependency definitions. |
| `frontend/package-lock.json` | Exact dependency lockfile for reproducible installs. |
| `frontend/vite.config.ts` | Vite build/dev server configuration. |
| `frontend/vite-env.d.ts` | Type declarations for Vite env variables. |
| `frontend/tsconfig.json` | TypeScript compiler settings for frontend project. |
| `frontend/eslint.config.mjs` | ESLint rules configuration file. |
| `frontend/postcss.config.js` | PostCSS pipeline config (Tailwind + autoprefixer). |
| `frontend/tailwind.config.ts` | Tailwind theme/content scanning configuration. |
| `frontend/index.html` | Base HTML page for mounting React app. |

### Frontend app (layout and pages)

| File | Purpose (1-2 lines) |
|---|---|
| `frontend/app/globals.css` | Global CSS utilities and base app styling. |
| `frontend/app/layout.tsx` | Shared root layout wrapper for all pages. |
| `frontend/app/providers.tsx` | App-level providers (React Query, theme, tooltip). |
| `frontend/app/page.tsx` | Landing/home overview page. |
| `frontend/app/dashboard/page.tsx` | Main prediction dashboard page with form + results. |
| `frontend/app/models/page.tsx` | Model comparison page with metrics charts/tables. |
| `frontend/app/what-if/page.tsx` | What-if simulation page for two profile comparison. |
| `frontend/app/history/page.tsx` | Prediction history listing and trend view. |
| `frontend/app/about/page.tsx` | "How it works" explanation page. |

### Frontend shared components

| File | Purpose (1-2 lines) |
|---|---|
| `frontend/components/app-shell.tsx` | Global header/nav shell used across pages. |
| `frontend/components/patient-form.tsx` | Reusable form renderer for mushroom input fields. |
| `frontend/components/risk-summary.tsx` | Main score/risk summary card component. |
| `frontend/components/recommendations-panel.tsx` | Shows recommendation list from backend output. |
| `frontend/components/top-factors-chart.tsx` | Chart for top contributing prediction factors. |
| `frontend/components/history-trend-chart.tsx` | Trend chart built from saved history records. |
| `frontend/components/refresh-indicator.tsx` | Displays last-refresh and fetching state in UI. |
| `frontend/components/health-indicator.tsx` | Shows backend health/availability status. |
| `frontend/components/theme-toggle.tsx` | Light/dark mode toggle control. |

### Frontend UI primitives (shadcn-style)

| File | Purpose (1-2 lines) |
|---|---|
| `frontend/components/ui/button.tsx` | Reusable button component styles and variants. |
| `frontend/components/ui/card.tsx` | Reusable card container components. |
| `frontend/components/ui/dialog.tsx` | Reusable modal dialog wrapper. |
| `frontend/components/ui/input.tsx` | Reusable text/number input field component. |
| `frontend/components/ui/label.tsx` | Reusable label component. |
| `frontend/components/ui/select.tsx` | Reusable dropdown/select component. |
| `frontend/components/ui/badge.tsx` | Small status badge/pill component. |
| `frontend/components/ui/skeleton.tsx` | Loading placeholder UI component. |
| `frontend/components/ui/table.tsx` | Reusable table layout components. |
| `frontend/components/ui/tooltip.tsx` | Reusable tooltip wrapper component. |

### Frontend hooks, libs, services, types, source

| File | Purpose (1-2 lines) |
|---|---|
| `frontend/hooks/use-risk-api.ts` | Query/mutation hooks for all backend API calls. |
| `frontend/lib/patient-config.ts` | Feature field metadata, groups, labels, defaults. |
| `frontend/lib/schemas.ts` | Zod schemas for form validation. |
| `frontend/lib/model-guidance.ts` | Simple "best use-case" text for each model. |
| `frontend/lib/mushroom-description.ts` | Builds plain-language paragraph from selected traits. |
| `frontend/lib/utils.ts` | Shared utility helpers (class merge, model formatting, risk color). |
| `frontend/services/api.ts` | Axios API client functions for backend endpoints. |
| `frontend/types/api.ts` | TypeScript interfaces for API request/response shapes. |
| `frontend/src/main.tsx` | React app entrypoint and root renderer. |
| `frontend/src/App.tsx` | Router mapping for all frontend routes/pages. |
| `frontend/public/logo.svg` | Static logo asset used by frontend. |
| `frontend/styles/theme.css` | Additional theme-level styling definitions. |

---

## 6) Quick Note
- "Source files" are files we write and maintain.
- "Generated files" like `.pyc`, `.db`, `.joblib` are created by runtime/training and are still important for app behavior.

