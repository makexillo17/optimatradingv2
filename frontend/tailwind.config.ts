import type { Config } from "tailwindcss";

const config: Config = {
    content: [
        "./app/**/*.{js,ts,jsx,tsx,mdx}",
        "./pages/**/*.{js,ts,jsx,tsx,mdx}",
        "./components/**/*.{js,ts,jsx,tsx,mdx}",
        "./src/**/*.{js,ts,jsx,tsx,mdx}",
    ],
    theme: {
        extend: {
            colors: {
                background: "#000000",
                "border-subtle": "#1A1A1A",
                "accent-primary": "#007AFF",
                "text-primary": "#EDEDED",
                "text-secondary": "#888888",
                "signal-up": "#10B981",
                "signal-down": "#F43F5E",
            },
            fontFamily: {
                sans: ["var(--font-inter)"],
                mono: ["var(--font-jetbrains-mono)"],
            },
            backgroundImage: {
                "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
            },
        },
    },
    plugins: [],
};
export default config;
