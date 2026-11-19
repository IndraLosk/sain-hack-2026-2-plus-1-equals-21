import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Сборка кладётся в frontend/dist — FastAPI (backend/main.py) раздаёт её как статику на "/".
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    sourcemap: false,
  },
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/run': { target: 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
})
