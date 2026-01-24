import { cn } from "@/utils/cn"; // Assuming simple utility provided or inline
// Since I can't guarantee utils/cn exists from my minimal setup, I will inline or use a simple helper or just clsx import if I assumed it installed. 
// I'll stick to bare import and assume user can fix or I'll provide a local helper if complex.
// Actually, I recommended installing clsx tailwind-merge in package.json.
// Let's create a definition for 'cn' inside the component or expect a utility file.
// To be safe and compliant with standard patterns, I'll assume standard usage but maybe inline strictly for this file to be standalone if utils doesn't exist.
// Prompt asked for specific output files. 'utils/cn.ts' was not asked. I will implement with basic implementation or import 'clsx' and 'twMerge' directly if I can't create extra files.
// But I can create extra files if needed for "working code".
// Let's just use clsx/tailwind-merge inline to be safe.

import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

function cn(...inputs: ClassValue[]) {
    return twMerge(clsx(inputs));
}

interface GlassCardProps extends React.HTMLAttributes<HTMLDivElement> {
    children: React.ReactNode;
    active?: boolean;
}

export default function GlassCard({ children, className, active, ...props }: GlassCardProps) {
    return (
        <div
            className={cn(
                "relative overflow-hidden rounded-xl border transition-all duration-300",
                "bg-white/5 backdrop-blur-md", // Base Glass Style
                "border-white/10",             // Subtle Border
                active ? "border-accent-primary/50 shadow-[0_0_20px_-10px_rgba(0,122,255,0.3)]" : "hover:border-white/20",
                className
            )}
            {...props}
        >
            {/* Optional: Noise texture or gradient overlay could go here */}
            <div className="relative z-10 h-full">
                {children}
            </div>
        </div>
    );
}
