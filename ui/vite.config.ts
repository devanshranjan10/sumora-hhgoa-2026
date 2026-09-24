import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // traces + answers are served from the repo root during dev
      '/data': 'http://localhost:5174',
    },
  },
})
