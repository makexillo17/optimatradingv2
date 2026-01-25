import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import { motion } from "framer-motion";

// Helper inline to avoid extra file dependency for this task
function cn(...inputs: ClassValue[]) {
    return twMerge(clsx(inputs));
}

export const BentoGrid = ({
    className,
    children,
}: {
    className?: string;
    children?: React.ReactNode;
}) => {
    return (
        <div
            className={cn(
                "grid md:auto-rows-[18rem] grid-cols-1 md:grid-cols-3 gap-4 max-w-7xl mx-auto ",
                className
            )}
        >
            {children}
        </div>
    );
};

export const BentoGridItem = ({
    className,
    title,
    description,
    header,
    icon,
}: {
    className?: string;
    title?: string | React.ReactNode;
    description?: string | React.ReactNode;
    header?: React.ReactNode;
    icon?: React.ReactNode;
}) => {
    return (
        <motion.div
            whileHover={{ scale: 1.01 }}
            className={cn(
                "row-span-1 rounded-xl group/bento hover:shadow-xl transition duration-200 shadow-input dark:shadow-none p-4 dark:bg-black dark:border-white/[0.1] bg-white border border-transparent justify-between flex flex-col space-y-4",
                "border-white/10 bg-black", // Force dark mode presets given User requirements
                className
            )}
        >
            <div className="group-hover/bento:translate-x-2 transition duration-200 h-full flex flex-col">
                {header}
                {(title || description || icon) && (
                    <div className="pt-4  transition duration-200 group-hover/bento:-translate-y-1">
                        {icon}
                        <div className="font-sans font-bold text-neutral-600 dark:text-neutral-200 mb-2 mt-2">
                            {title}
                        </div>
                        <div className="font-sans font-normal text-neutral-600 text-xs dark:text-neutral-300">
                            {description}
                        </div>
                    </div>
                )}
            </div>
        </motion.div>
    );
};
