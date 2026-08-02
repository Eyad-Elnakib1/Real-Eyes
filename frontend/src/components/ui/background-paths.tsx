"use client";

import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import Link from "next/link";

function FloatingPaths({ position }: { position: number }) {
    const paths = Array.from({ length: 36 }, (_, i) => ({
        id: i,
        d: `M-${380 - i * 5 * position} -${189 + i * 6}C-${380 - i * 5 * position
            } -${189 + i * 6} -${312 - i * 5 * position} ${216 - i * 6} ${152 - i * 5 * position
            } ${343 - i * 6}C${616 - i * 5 * position} ${470 - i * 6} ${684 - i * 5 * position
            } ${875 - i * 6} ${684 - i * 5 * position} ${875 - i * 6}`,
        color: `rgba(255,255,255,${0.2 + i * 0.02})`,
        width: 0.6 + i * 0.02,
    }));

    return (
        <div className="absolute inset-0 pointer-events-none">
            <svg
                className="w-full h-full text-white"
                viewBox="0 0 696 316"
                fill="none"
            >
                <title>Background Paths</title>
                {paths.map((path) => (
                    <motion.path
                        key={path.id}
                        d={path.d}
                        stroke="currentColor"
                        strokeWidth={path.width}
                        strokeOpacity={0.188}
                        initial={{ pathLength: 0.3, opacity: 0.6 }}
                        animate={{
                            pathLength: 1,
                            opacity: [0.3, 0.6, 0.3],
                            pathOffset: [0, 1, 0],
                        }}
                        transition={{
                            duration: 20 + Math.random() * 10,
                            repeat: Number.POSITIVE_INFINITY,
                            ease: "linear",
                        }}
                    />
                ))}
            </svg>
        </div>
    );
}

export function BackgroundPaths({
    title = "Background Paths",
}: {
    title?: string;
}) {
    const words = title.split(" ");

    return (
        <div className="relative min-h-screen w-full flex items-center justify-center overflow-hidden bg-neutral-950">
            <div
                className="absolute inset-0"
                style={{
                    maskImage: "radial-gradient(circle at center, transparent 30%, black 60%)",
                    WebkitMaskImage: "radial-gradient(circle at center, transparent 30%, black 60%)"
                }}
            >
                <FloatingPaths position={1} />
                <FloatingPaths position={-1} />
            </div>

            <div className="relative z-10 container mx-auto px-4 md:px-6 text-center">
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 2 }}
                    className="max-w-4xl mx-auto flex flex-col items-center justify-center min-h-screen w-full relative overflow-hidden"
                >
                    <div className="relative flex flex-col items-center justify-center">
                        <h1
                            className="text-[11rem] font-extrabold tracking-tighter text-nowrap mask-b-from-20% mask-b-to-80%"
                            style={{ perspective: "1000px" }}
                        >
                            {"Welcome to".split(" ").map((word, wordIndex) => (
                                <span
                                    key={wordIndex}
                                    className="inline-block mr-4 last:mr-0"
                                >
                                    {word.split("").map((letter, letterIndex) => (
                                        <motion.span
                                            key={`${wordIndex}-${letterIndex}`}
                                            initial={{ y: 100, opacity: 0, rotateX: -90 }}
                                            animate={{ y: 0, opacity: 1, rotateX: 0 }}
                                            transition={{
                                                delay:
                                                    wordIndex * 0.1 +
                                                    letterIndex * 0.03,
                                                type: "spring",
                                                stiffness: 150,
                                                damping: 25,
                                            }}
                                            className="inline-block text-neutral-900"
                                            style={{
                                                textShadow: `
                                                    0 1px 0 #1a1a1a,
                                                    0 2px 0 #1f1f1f,
                                                    0 3px 0 #242424,
                                                    0 4px 0 #292929,
                                                    0 5px 0 #2e2e2e,
                                                    0 6px 0 #333333,
                                                    0 7px 0 #383838,
                                                    0 8px 0 #3d3d3d,
                                                    0 9px 1px rgba(0,0,0,.5),
                                                    0 0 10px rgba(0,0,0,.7),
                                                    0 5px 15px rgba(0,0,0,.9),
                                                    0 15px 30px rgba(0,0,0,.9),
                                                    0 30px 45px rgba(0,0,0,.9)
                                                `,
                                                transformStyle: "preserve-3d"
                                            }}
                                        >
                                            {letter}
                                        </motion.span>
                                    ))}
                                </span>
                            ))}
                        </h1>

                        <h1
                            className="text-[12rem] font-extrabold tracking-tighter text-nowrap mask-b-from-20% mask-b-to-80% -mt-16 z-10"
                            style={{ perspective: "1000px" }}
                        >
                            <span className="inline-block">
                                {"RealEyes".split("").map((letter, letterIndex) => (
                                    <motion.span
                                        key={`edcom-${letterIndex}`}
                                        initial={{ y: 100, opacity: 0, rotateX: -90 }}
                                        animate={{ y: 0, opacity: 1, rotateX: 0 }}
                                        transition={{
                                            delay:
                                                0.3 + letterIndex * 0.03,
                                            type: "spring",
                                            stiffness: 150,
                                            damping: 25,
                                        }}
                                        className="inline-block bg-gradient-to-b from-neutral-600 to-neutral-900 bg-clip-text text-transparent"
                                        style={{
                                            textShadow: letter === "E" ? `
                                                0 1px 0 #1a1a1a,
                                                0 2px 0 #1f1f1f,
                                                0 3px 0 #242424,
                                                0 4px 0 #292929,
                                                0 5px 0 #2e2e2e,
                                                0 6px 0 #333333,
                                                0 7px 0 #383838,
                                                0 8px 0 #3d3d3d,
                                                0 9px 0 #424242,
                                                0 10px 0 #474747,
                                                0 11px 0 #4c4c4c,
                                                0 12px 0 #515151,
                                                0 13px 0 #565656,
                                                0 14px 0 #5b5b5b,
                                                0 15px 0 #606060,
                                                0 16px 0 #656565,
                                                0 50px 50px rgba(0,0,0,.6),
                                                0 20px 80px rgba(0,0,0,.8),
                                                0 80px 120px rgba(0,0,0,.9)
                                            ` : letter === "D" ? `
                                                0 1px 0 #1a1a1a,
                                                0 2px 0 #1f1f1f,
                                                0 3px 0 #242424,
                                                0 4px 0 #292929,
                                                0 5px 0 #2e2e2e,
                                                0 6px 0 #333333,
                                                0 7px 0 #383838,
                                                0 8px 0 #3d3d3d,
                                                0 9px 0 #424242,
                                                0 10px 0 #474747,
                                                0 11px 0 #4c4c4c,
                                                0 12px 0 #515151,
                                                0 13px 0 #565656,
                                                0 14px 0 #5b5b5b,
                                                0 15px 0 #606060,
                                                0 16px 0 #656565,
                                                0 17px 0 #6a6a6a,
                                                0 18px 0 #6f6f6f,
                                                0 50px 50px rgba(0,0,0,.6),
                                                0 20px 80px rgba(0,0,0,.8),
                                                0 80px 120px rgba(0,0,0,.9)
                                            ` : `
                                                0 1px 0 #1a1a1a,
                                                0 2px 0 #1f1f1f,
                                                0 3px 0 #242424,
                                                0 4px 0 #292929,
                                                0 5px 0 #2e2e2e,
                                                0 6px 0 #333333,
                                                0 7px 0 #383838,
                                                0 8px 0 #3d3d3d,
                                                0 9px 0 #424242,
                                                0 10px 0 #474747,
                                                0 11px 0 #4c4c4c,
                                                0 12px 0 #515151,
                                                0 13px 0 #565656,
                                                0 14px 0 #5b5b5b,
                                                0 15px 0 #606060,
                                                0 16px 0 #656565,
                                                0 17px 0 #6a6a6a,
                                                0 18px 0 #6f6f6f,
                                                0 19px 0 #747474,
                                                0 20px 0 #797979,
                                                0 50px 50px rgba(0,0,0,.6),
                                                0 20px 80px rgba(0,0,0,.8),
                                                0 80px 120px rgba(0,0,0,.9)
                                            `,
                                            transformStyle: "preserve-3d"
                                        }}
                                    >
                                        {letter}
                                    </motion.span>
                                ))}
                            </span>
                        </h1>
                    </div>


                    <div
                        className="inline-block group relative bg-neutral-900/10 
                        dark:from-white/10 dark:to-black/10 p-px rounded-2xl backdrop-blur-lg 
                        overflow-hidden shadow-lg hover:shadow-xl transition-shadow duration-300 -mt-8"
                    >
                        <Link href="/dashboard">
                            <Button
                                variant="ghost"
                                className="rounded-[1.15rem] px-8 py-6 text-lg font-semibold backdrop-blur-md 
                            bg-neutral-900/95 hover:bg-neutral-800/100 dark:bg-black/95 dark:hover:bg-black/100 
                            text-white transition-all duration-300 
                            group-hover:-translate-y-0.5 border border-white/10 dark:border-white/10
                            hover:shadow-md dark:hover:shadow-neutral-800/50 flex gap-2 items-center"
                            >
                                <span className="opacity-90 group-hover:opacity-100 transition-opacity">
                                    Discover RealEyes
                                </span>
                                <span
                                    className="ml-3 opacity-70 group-hover:opacity-100 group-hover:translate-x-1.5 
                                transition-all duration-300 size-4 mr-2"
                                    data-icon="inline-start"
                                >
                                    →
                                </span>
                            </Button>
                        </Link>
                    </div>
                </motion.div>
            </div>
        </div>
    );
}
