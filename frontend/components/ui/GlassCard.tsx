import React from "react";
import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

function cn(...inputs: (string | undefined | null | false)[]) {
  return twMerge(clsx(inputs));
}

interface GlassCardProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
  active?: boolean;
}

export default function GlassCard({
  children,
  className,
  active,
  ...props
}: GlassCardProps) {
  return (
    <div
      className={cn(
        "relative overflow-hidden rounded-xl border transition-all duration-300",
        "bg-white/5 backdrop-blur-md", // Base Glass Style
        "border-white/10",             // Subtle Border
        active
          ? "border-accent-primary/50 shadow-[0_0_20px_-10px_rgba(0,122,255,0.3)] bg-accent-primary/5"
          : "hover:border-white/20",
        className
      )}
      {...props}
    >
      <div className="relative z-10 h-full">
        {children}
      </div>
    </div>
  );
}
