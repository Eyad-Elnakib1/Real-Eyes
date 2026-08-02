"use client";

import * as React from "react";
import { GooeyText } from "@/components/ui/gooey-text-morphing";
import { BackgroundPaths } from "@/components/ui/background-paths";
import { motion, AnimatePresence } from "framer-motion";

export default function Home() {
  const [showBackground, setShowBackground] = React.useState(false);
  const [isExploding, setIsExploding] = React.useState(false);

  const handleWordChange = (index: number) => {
    // index 6 is "Compare" (start=3, +3 updates = 6)
    if (index === 8) {
      // Let Compare stay for a second before exploding
      setTimeout(() => {
        setIsExploding(true);
      }, 900);

      // Transition to background paths
      setTimeout(() => {
        setShowBackground(true);
      }, 900);
    }
  };

  return (
    <main className="relative min-h-screen overflow-hidden bg-black">
      <AnimatePresence mode="wait">
        {!showBackground ? (
          <motion.div
            key="gooey-container"
            initial={{ opacity: 1 }}
            exit={{ opacity: 0, scale: 5, filter: "blur(20px)" }}
            transition={{ duration: 1.5, ease: "easeInOut" }}
            className="h-screen flex items-center justify-center bg-black"
          >
            <motion.div
              animate={isExploding ? { scale: 3, opacity: 0, filter: "blur(10px)" } : {}}
              transition={{ duration: 1.5, ease: "easeIn" }}
            >
              <GooeyText
                texts={["REAL","or", "Fake ", "The Answer in",  "RealEyes"]}
                morphTime={0.8}
                cooldownTime={0.3}
                className="font-bold relative z-50"
                textClassName="text-neutral-950"
                onWordChange={handleWordChange}
              />
            </motion.div>
          </motion.div>
        ) : (
          <motion.div
            key="background-paths"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.1 }}
          >
            <BackgroundPaths title="Welcome to RealEyes" />
          </motion.div>
        )}
      </AnimatePresence>
    </main>
  );
}
