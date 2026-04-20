import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const proxyTarget = env.VITE_AGENT_PROXY_TARGET || 'http://127.0.0.1:8020';

  return {
    plugins: [react()],
    server: {
      host: '127.0.0.1',
      proxy: {
        '/api/v1/agent': {
          target: proxyTarget,
          changeOrigin: true,
        },
      },
    },
  };
});
