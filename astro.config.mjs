import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import tailwind from "@astrojs/tailwind";

// https://astro.build/config
export default defineConfig({
  site: 'https://pantagruelalpha.github.io',
  base: '/pantagruel-research',
  integrations: [mdx(), sitemap(), tailwind()]
});