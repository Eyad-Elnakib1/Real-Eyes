import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'placehold.co',
      },
      {
        protocol: 'https',
        hostname: 'via.placeholder.com'
      }
    ],
  },
  experimental: {
    // optimizeCss: true, // Task 10.28: CSS Optimization (Disabled due to incompatibility with Tailwind v4 syntax)
  },
};

export default nextConfig;
