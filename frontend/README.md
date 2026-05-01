# Frontend: Smart Mushroom Toxicity Analysis System

## Stack
- Vite + React + TypeScript
- React Router
- Tailwind CSS
- shadcn/ui (Radix)
- TanStack Query
- React Hook Form + Zod
- Recharts + Framer Motion

## Setup
```bash
cd frontend
npm install
```

## Configure Backend URL
Create `frontend/.env.local`:
```bash
VITE_API_BASE_URL=http://localhost:8000/api
```

## Run
```bash
cd frontend
npm run dev
```

## Build
```bash
npm run build
npm run start
```

## Main Routes
- `/` Project overview and live status
- `/dashboard` Main toxicity prediction form
- `/models` Model comparison and best-use guidance
- `/what-if` Side-by-side profile simulation
- `/history` Saved predictions and trend
- `/about` Dataset and training pipeline details
