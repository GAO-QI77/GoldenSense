# Deployment Guide 🌍

Follow these steps to deploy your site to popular hosting platforms.

## Option 1: Vercel (Recommended)
1. Push your code to GitHub.
2. Go to [Vercel.com](https://vercel.com).
3. Click "New Project" and import your repository.
4. Vercel will automatically detect Vite. Click "Deploy".

## Option 2: Netlify
1. Login to [Netlify.com](https://netlify.com).
2. Choose "Import from git".
3. Select your repository.
4. Build settings:
   - Build command: `npm run build`
   - Publish directory: `dist`
5. Click "Deploy site".

## Option 3: GitHub Pages
1. Install the `gh-pages` package:
   ```bash
   npm install gh-pages --save-dev
   ```
2. Add the following to `package.json`:
   ```json
   "homepage": "https://yourusername.github.io/your-repo-name",
   "scripts": {
     "predeploy": "npm run build",
     "deploy": "gh-pages -d dist"
   }
   ```
3. Run `npm run deploy`.

## Local Preview
To test the production build locally before deploying:
```bash
npm run build
npm run preview
```
