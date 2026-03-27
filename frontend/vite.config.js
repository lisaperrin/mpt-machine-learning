import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [sveltekit()],
	server: {
		proxy: {
			'/api': {
				target: 'http://localhost:8003',
				changeOrigin: true,
				secure: false
			}
		}
	},
	preview: {
		proxy: {
			'/api': {
				target: 'http://localhost:8003',
				changeOrigin: true,
				secure: false
			}
		}
	}
});