"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { UploadCloud, FileText, Image as ImageIcon, ShieldCheck, AlertTriangle, ShieldAlert, CheckCircle2, Layout, Video } from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:5001";

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState("text");
  
  // API connection status: connected (live Flask server) or demo (fallback to simulated responses)
  const [apiStatus, setApiStatus] = useState<"demo" | "connected">("demo");

  // States for text analysis
  const [textInput, setTextInput] = useState("");
  const [isTextLoading, setIsTextLoading] = useState(false);
  const [textResult, setTextResult] = useState<any>(null);

  // States for image analysis
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isImageLoading, setIsImageLoading] = useState(false);
  const [imageResult, setImageResult] = useState<any>(null);
  const [imageAnalysisMode, setImageAnalysisMode] = useState<"full" | "seg_only">("full");
  const [hoveredMode, setHoveredMode] = useState<"full" | "seg_only" | null>(null);
  const [imageResultView, setImageResultView] = useState<"classification" | "segmentation">("segmentation");

  // States for post analysis
  const [postTextInput, setPostTextInput] = useState("");
  const [postImageFile, setPostImageFile] = useState<File | null>(null);
  const [postImagePreview, setPostImagePreview] = useState<string | null>(null);
  const [isPostLoading, setIsPostLoading] = useState(false);
  const [postResult, setPostResult] = useState<any>(null);
  const [postResultView, setPostResultView] = useState<"overall" | "text" | "image">("overall");

  // States for video analysis
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoPreview, setVideoPreview] = useState<string | null>(null);
  const [isVideoLoading, setIsVideoLoading] = useState(false);
  const [videoResult, setVideoResult] = useState<any>(null);

  // Zoomable heatmap modal state
  const [selectedReportUrl, setSelectedReportUrl] = useState<string | null>(null);

  // Verify connection to the local Python Flask server (/health) on mount
  useEffect(() => {
    const checkConnection = async () => {
      try {
        const res = await fetch(`${API_BASE}/health`);
        const data = await res.json();
        if (data.ok) {
          setApiStatus("connected");
        }
      } catch (e) {
        setApiStatus("demo");
      }
    };
    checkConnection();
  }, []);

  const handleTextAnalyze = async () => {
    if (!textInput.trim()) return;
    setIsTextLoading(true);
    setTextResult(null);

    try {
      const startRes = await fetch(`${API_BASE}/verify-text-start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: textInput }),
      });

      if (startRes.ok) {
        const startData = await startRes.json();
        if (startData.no_claim) {
          setTextResult({
            verdict: "No Claim Detected",
            confidence: 0.0,
            explanation: "The text input does not seem to declare a fact-checkable claim or is too subjective.",
            sources: []
          });
          setIsTextLoading(false);
          return;
        }

        const taskId = startData.task_id;
        const pollInterval = setInterval(async () => {
          try {
            const pollRes = await fetch(`${API_BASE}/verify-progress/${taskId}`);
            const pollData = await pollRes.json();
            if (pollData.done) {
              clearInterval(pollInterval);
              if (pollData.error) {
                runMockTextAnalysis();
              } else {
                const resObj = pollData.result;
                setTextResult({
                  verdict: resObj.final_prediction === "SUPPORTS" ? "Verified" : resObj.final_prediction === "REFUTES" ? "Misinformation" : "Not Enough Info",
                  confidence: resObj.confidence,
                  explanation: resObj.explanation || resObj.note || "No explanation provided.",
                  sources: resObj.evidence || []
                });
                setApiStatus("connected");
                setIsTextLoading(false);
              }
            }
          } catch (err) {
            clearInterval(pollInterval);
            runMockTextAnalysis();
          }
        }, 2000);
        return;
      }
    } catch (err) {
      // Server offline, use mock
    }

    runMockTextAnalysis();
  };

  const runMockTextAnalysis = () => {
    setTimeout(() => {
      setTextResult({
        verdict: "Misinformation",
        confidence: 0.85,
        explanation: "This claim lacks credible supporting evidence and contradicts reports from multiple reputable news organizations. Be cautious before sharing. (Simulated Demo Data)",
        sources: [
          { title: "Reuters Fact Check", url: "#" },
          { title: "Snopes Analysis", url: "#" }
        ]
      });
      setApiStatus("demo");
      setIsTextLoading(false);
    }, 2000);
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setImageFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result as string);
      };
      reader.readAsDataURL(file);
      setImageResult(null);
    }
  };

  const handleImageAnalyze = async () => {
    if (!imagePreview) return;
    setIsImageLoading(true);
    setImageResult(null);

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          imageUrl: imagePreview,
          forceSeg: imageAnalysisMode === "seg_only"
        }),
      });

      if (res.ok) {
        const data = await res.json();
        if (data.ok) {
          setImageResult({
            verdict: (data.label === "AI Generated" || data.label === "FAKE" || data.pred === 1) ? "AI Generated" : "Authentic Image",
            confidence: data.probs ? (data.pred === 1 ? data.probs[1] : data.probs[0]) : 0.90,
            classificationUrl: `${API_BASE}/report?path=${encodeURIComponent(data.out_path)}`,
            segmentationUrl: data.seg_path ? `${API_BASE}/report?path=${encodeURIComponent(data.seg_path)}` : null,
          });
          setImageResultView(data.seg_path ? "segmentation" : "classification");
          setApiStatus("connected");
          setIsImageLoading(false);
          return;
        }
      }
    } catch (err) {
      // Server offline, use mock
    }

    // Mock response
    setTimeout(() => {
      setImageResult({
        verdict: "AI Generated",
        confidence: 0.92,
        classificationUrl: imagePreview,
        segmentationUrl: imagePreview,
      });
      setImageResultView("segmentation");
      setApiStatus("demo");
      setIsImageLoading(false);
    }, 2500);
  };

  const handlePostImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setPostImageFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPostImagePreview(reader.result as string);
      };
      reader.readAsDataURL(file);
      setPostResult(null);
      setPostResultView("overall");
    }
  };

  const handlePostAnalyze = async () => {
    if (!postImagePreview) return;
    setIsPostLoading(true);
    setPostResult(null);
    setPostResultView("overall");

    try {
      const res = await fetch(`${API_BASE}/process-screenshot`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataUrl: postImagePreview }),
      });

      if (res.ok) {
        const data = await res.json();
        if (data.ok) {
          const regions = data.regions || [];
          const textResults = data.text_results || [];

          const hasFakeImage = regions.some((r: any) => r.label.toLowerCase().includes("fake") || r.label.toLowerCase().includes("ai"));
          const hasFakeText = textResults.some((t: any) => t.result && t.result.final_prediction === "REFUTES");

          let verdict = "Authentic Post";
          if (hasFakeImage && hasFakeText) verdict = "Manipulated Post";
          else if (hasFakeImage) verdict = "AI Generated Media";
          else if (hasFakeText) verdict = "Misleading Content";

          setPostResult({
            verdict,
            confidence: 0.85,
            explanation: `Analysis found ${regions.length} sub-images and ${textResults.length} text sections.`,
            regions: regions.map((r: any) => ({
              id: r.id,
              type: r.type,
              label: r.label,
              confidence: r.probs ? (r.label.toLowerCase().includes("real") ? r.probs[0] : r.probs[1]) : 0.8,
              classificationUrl: r.out_path ? `${API_BASE}/report?path=${encodeURIComponent(r.out_path)}` : null,
              segmentationUrl: r.seg_path ? `${API_BASE}/report?path=${encodeURIComponent(r.seg_path)}` : null,
            })),
            textResults: textResults.map((t: any) => ({
              source: t.source,
              raw: t.raw,
              noClaim: t.no_claim,
              verdict: t.result ? (t.result.final_prediction === "SUPPORTS" ? "Verified" : t.result.final_prediction === "REFUTES" ? "Misinformation" : "Not Enough Info") : "No Claim",
              explanation: t.result ? (t.result.explanation || "No details provided.") : "No checkable factual claim found in this section.",
              sources: t.result ? (t.result.evidence || []) : [],
              confidence: t.result ? (t.result.confidence || 0.8) : 0.0
            }))
          });
          setApiStatus("connected");
          setIsPostLoading(false);
          return;
        }
      }
    } catch (err) {
      // Server offline, use mock
    }

    // Mock fallback
    setTimeout(() => {
      setPostResult({
        verdict: "Manipulated Post",
        confidence: 0.88,
        explanation: "The text contains misleading claims, and the attached image has signs of AI generation. This combination is highly indicative of synthetic propaganda. (Simulated Demo Data)",
        textConfidence: 0.75,
        imageConfidence: 0.94,
        regions: [
          { 
            id: 1, 
            type: "image block", 
            label: "AI Generated", 
            confidence: 0.94, 
            classificationUrl: postImagePreview,
            segmentationUrl: postImagePreview
          }
        ],
        textResults: [
          { source: "outside", raw: "Breaking: New scientific study confirms flat earth theory.", noClaim: false, verdict: "Misinformation", explanation: "Flat earth claims are scientifically refuted. All space agencies and physics models confirm the Earth's geoid shape.", sources: [], confidence: 0.85 }
        ]
      });
      setApiStatus("demo");
      setIsPostLoading(false);
    }, 3000);
  };

  const handleVideoUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      // Check 12 MB limit
      if (file.size > 12 * 1024 * 1024) {
        alert("Video file exceeds the 12 MB size limit.");
        return;
      }
      setVideoFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setVideoPreview(reader.result as string);
      };
      reader.readAsDataURL(file);
      setVideoResult(null);
    }
  };

  const handleVideoAnalyze = async () => {
    if (!videoFile) return;
    setIsVideoLoading(true);
    setVideoResult(null);

    try {
      const formData = new FormData();
      formData.append("video", videoFile);
      const res = await fetch(`${API_BASE}/classify-video`, {
        method: "POST",
        body: formData,
      });

      if (res.ok) {
        const data = await res.json();
        if (data.ok) {
          setVideoResult({
            verdict: data.probs[1] > 0.5 ? "Deepfake Detected" : "Verified Video",
            confidence: data.probs[1] > 0.5 ? data.probs[1] : data.probs[0],
            explanation: data.explanation || "Video frame analysis completed using Swin V2-B model over 12 sampled frames."
          });
          setApiStatus("connected");
          setIsVideoLoading(false);
          return;
        }
      }
    } catch (err) {
      // Offline fallback
    }

    setTimeout(() => {
      setVideoResult({
        verdict: "Deepfake Detected",
        confidence: 0.95,
        explanation: "Audio-visual inconsistencies and facial artifact analysis strongly suggest this is an AI-generated deepfake video. (Simulated Demo Data)",
      });
      setApiStatus("demo");
      setIsVideoLoading(false);
    }, 3500);
  };

  const renderVerdictBadge = (verdict: string) => {
    let color = "bg-green-500/10 text-green-400 border-green-500/30 shadow-[0_0_15px_rgba(34,197,94,0.3)]";
    let Icon = CheckCircle2;
    const vLower = verdict.toLowerCase();
    if (vLower.includes("fake") || vLower.includes("misinformation") || vLower.includes("ai generated") || vLower.includes("manipulated") || vLower.includes("misleading") || vLower.includes("refutes")) {
      color = "bg-red-500/10 text-red-400 border-red-500/30 shadow-[0_0_15px_rgba(239,68,68,0.3)]";
      Icon = AlertTriangle;
    } else if (vLower.includes("info") || vLower.includes("no claim")) {
      color = "bg-yellow-500/10 text-yellow-400 border-yellow-500/30 shadow-[0_0_15px_rgba(234,179,8,0.3)]";
      Icon = ShieldCheck;
    }

    return (
      <Badge variant="outline" className={`px-3 py-1.5 text-sm font-medium flex gap-2 items-center rounded-full ${color}`}>
        <Icon className="w-4 h-4" />
        {verdict}
      </Badge>
    );
  };

  const isLoading = 
    (activeTab === "text" && isTextLoading) || 
    (activeTab === "image" && isImageLoading) || 
    (activeTab === "post" && isPostLoading) || 
    (activeTab === "video" && isVideoLoading);

  return (
    <div className="min-h-screen bg-slate-950 text-neutral-100 flex flex-col relative overflow-hidden">
      {/* Dynamic Glowing Orbs Background */}
      <div className="absolute inset-0 z-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-20%] right-[-10%] w-[800px] h-[800px] rounded-full bg-violet-600/20 blur-[120px] opacity-60 mix-blend-screen animate-float-slow" />
        <div className="absolute bottom-[-20%] left-[-10%] w-[600px] h-[600px] rounded-full bg-cyan-600/20 blur-[120px] opacity-60 mix-blend-screen animate-float-reverse" />
        <div className="absolute top-[40%] left-[40%] w-[400px] h-[400px] rounded-full bg-blue-600/10 blur-[100px] opacity-40 mix-blend-screen animate-pulse" />
      </div>

      {/* Header */}
      <header className="relative z-10 border-b border-white/10 bg-black/50 backdrop-blur-xl px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <img src="/logo.png" alt="RealEyes Logo" className="w-10 h-10 object-contain rounded-full shadow-lg" />
          <h1 className="text-xl font-heading font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-white/60">
            RealEyes
          </h1>
        </div>

        {/* Pulsing connection indicator */}
        <div className="flex items-center gap-2 bg-neutral-950/60 border border-white/10 px-3.5 py-1.5 rounded-full backdrop-blur-md shadow-inner">
          <span className="relative flex h-2 w-2">
            <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${apiStatus === "connected" ? "bg-green-400" : "bg-amber-400"}`}></span>
            <span className={`relative inline-flex rounded-full h-2 w-2 ${apiStatus === "connected" ? "bg-green-500" : "bg-amber-500"}`}></span>
          </span>
          <span className="text-xs font-mono tracking-tight text-neutral-400">
            {apiStatus === "connected" ? "Live API" : "Demo Mode"}
          </span>
        </div>
      </header>

      {/* Main Content */}
      <main className="relative z-10 flex-1 container max-w-7xl mx-auto p-6 md:p-12 flex flex-col gap-8">
        <div className="flex flex-col gap-2">
          <h2 className="text-3xl md:text-4xl font-heading font-bold text-white tracking-tight">Fact-Check Anything</h2>
          <p className="text-neutral-400 text-lg max-w-2xl">
            Analyze text, images, posts, or video for misinformation or AI generation. RealEyes brings truth to focus.
          </p>
        </div>

        <Tabs defaultValue="text" value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-4 max-w-3xl bg-neutral-900/40 border border-white/5 p-1.5 backdrop-blur-xl rounded-xl shadow-inner">
            {["text", "image", "post", "video"].map((tab) => {
              const icons = {
                text: <FileText className="w-4 h-4 mr-2 hidden md:block" />,
                image: <ImageIcon className="w-4 h-4 mr-2 hidden md:block" />,
                post: <Layout className="w-4 h-4 mr-2 hidden md:block" />,
                video: <Video className="w-4 h-4 mr-2 hidden md:block" />
              };
              const labels = { text: "Text", image: "Image", post: "Post", video: "Video" };
              return (
                <TabsTrigger 
                  key={tab} 
                  value={tab} 
                  className="relative rounded-lg transition-all text-xs md:text-sm data-[state=active]:text-white data-[state=inactive]:text-neutral-400 z-10 py-2.5 outline-none hover:text-neutral-200"
                >
                  {activeTab === tab && (
                    <motion.div 
                      layoutId="activeTab" 
                      className="absolute inset-0 bg-slate-800 rounded-lg -z-10 shadow-lg border border-white/10" 
                      transition={{ type: "spring", bounce: 0.2, duration: 0.6 }} 
                    />
                  )}
                  <div className="flex items-center justify-center relative z-20 font-medium">
                    {icons[tab as keyof typeof icons]}
                    {labels[tab as keyof typeof labels]}
                  </div>
                </TabsTrigger>
              );
            })}
          </TabsList>

          <div className="mt-8 grid grid-cols-1 lg:grid-cols-12 gap-8">
            {/* Input Column */}
            <div className="flex flex-col gap-6 lg:col-span-5">
              <AnimatePresence mode="wait">
                {activeTab === "text" && (
                  <motion.div
                    key="text-input"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    transition={{ duration: 0.3 }}
                  >
                    <Card className="bg-slate-900/40 border-white/10 border-t-white/20 backdrop-blur-2xl shadow-[0_8px_32px_0_rgba(0,0,0,0.5)]">
                      <CardHeader>
                        <CardTitle className="text-xl">Analyze Claim</CardTitle>
                        <CardDescription className="text-neutral-400">
                          Paste a news excerpt, tweet, or any text claim below.
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        <Textarea 
                          placeholder="Paste text here to fact-check..." 
                          className="h-[350px] overflow-y-auto bg-black/50 border-white/10 resize-none text-base placeholder:text-neutral-600 focus-visible:ring-1 focus-visible:ring-neutral-700 text-white glass-scrollbar"
                          value={textInput}
                          onChange={(e) => setTextInput(e.target.value)}
                        />
                      </CardContent>
                      <CardFooter>
                        <Button 
                          onClick={handleTextAnalyze} 
                          disabled={!textInput.trim() || isTextLoading}
                          className="w-full bg-white text-black hover:bg-neutral-200 hover:scale-[1.02] active:scale-[0.98] shadow-[0_0_20px_rgba(255,255,255,0.1)] transition-all duration-300 font-semibold rounded-lg py-6"
                        >
                          {isTextLoading ? "Analyzing..." : "Analyze Text"}
                        </Button>
                      </CardFooter>
                    </Card>
                  </motion.div>
                )}

                {activeTab === "image" && (
                  <motion.div
                    key="image-input"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    transition={{ duration: 0.3 }}
                  >
                    <Card className="bg-slate-900/40 border-white/10 border-t-white/20 backdrop-blur-2xl shadow-[0_8px_32px_0_rgba(0,0,0,0.5)]">
                      <CardHeader>
                        <CardTitle className="text-xl">Detect AI Image</CardTitle>
                        <CardDescription className="text-neutral-400">
                          Upload an image to check if it was generated by AI.
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        <div className="flex flex-col gap-4">
                          <Label htmlFor="image-upload" className="cursor-pointer group">
                            <div className={`border-2 border-dashed rounded-xl flex flex-col items-center justify-center p-8 transition-all ${imagePreview ? 'border-neutral-700 bg-black/30' : 'border-neutral-700 hover:border-neutral-500 bg-neutral-950/50 hover:bg-neutral-900/50'}`}>
                              {imagePreview ? (
                                <div className="relative w-full aspect-video rounded-lg overflow-hidden flex items-center justify-center bg-black/50">
                                  <img src={imagePreview} alt="Preview" className="max-h-full object-contain" />
                                </div>
                              ) : (
                                <div className="flex flex-col items-center gap-3 text-neutral-400 group-hover:text-neutral-300">
                                  <UploadCloud className="w-10 h-10" />
                                  <span className="font-medium">Click to upload or drag and drop</span>
                                  <span className="text-xs text-neutral-500">PNG, JPG or WEBP (max. 5MB)</span>
                                </div>
                              )}
                            </div>
                            <Input 
                              id="image-upload" 
                              type="file" 
                              accept="image/*" 
                              className="hidden" 
                              onChange={handleImageUpload}
                            />
                          </Label>

                          {/* Analysis Mode Switch */}
                          {imagePreview && (
                            <div className="space-y-2.5 pt-2 border-t border-white/5">
                              <Label className="text-sm font-semibold text-neutral-300">Analysis Mode</Label>
                              <div className="grid grid-cols-2 gap-3 bg-black/40 p-1 rounded-xl border border-white/5">
                                <button
                                  type="button"
                                  onClick={() => setImageAnalysisMode("full")}
                                  onMouseEnter={() => setHoveredMode("full")}
                                  onMouseLeave={() => setHoveredMode(null)}
                                  className={`py-2 px-3 rounded-lg text-xs font-semibold transition-all ${imageAnalysisMode === "full" ? "bg-neutral-800 text-white shadow-md" : "text-neutral-500 hover:text-neutral-300"}`}
                                >
                                  Full Analysis (Auto)
                                </button>
                                <button
                                  type="button"
                                  onClick={() => setImageAnalysisMode("seg_only")}
                                  onMouseEnter={() => setHoveredMode("seg_only")}
                                  onMouseLeave={() => setHoveredMode(null)}
                                  className={`py-2 px-3 rounded-lg text-xs font-semibold transition-all ${imageAnalysisMode === "seg_only" ? "bg-neutral-800 text-white shadow-md" : "text-neutral-500 hover:text-neutral-300"}`}
                                >
                                  Detect the part
                                </button>
                              </div>
                              <p className="text-[11px] text-neutral-400 leading-normal min-h-[32px] transition-colors duration-150">
                                {(hoveredMode || imageAnalysisMode) === "full" 
                                  ? "Check if the photo is real or fake, and highlight modified parts if any are found." 
                                  : "Directly highlight the modified parts of the photo, skipping the real-or-fake check."
                                }
                              </p>
                            </div>
                          )}
                        </div>
                      </CardContent>
                      <CardFooter>
                        <Button 
                          onClick={handleImageAnalyze} 
                          disabled={!imageFile || isImageLoading}
                          className="w-full bg-white text-black hover:bg-neutral-200 hover:scale-[1.02] active:scale-[0.98] shadow-[0_0_20px_rgba(255,255,255,0.1)] transition-all duration-300 font-semibold rounded-lg py-6"
                        >
                          {isImageLoading 
                            ? "Analyzing..." 
                            : imageAnalysisMode === "full" 
                              ? "Analyze Image" 
                              : "Generate Heatmap"
                          }
                        </Button>
                      </CardFooter>
                    </Card>
                  </motion.div>
                )}

                {activeTab === "post" && (
                  <motion.div
                    key="post-input"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    transition={{ duration: 0.3 }}
                  >
                    <Card className="bg-slate-900/40 border-white/10 border-t-white/20 backdrop-blur-2xl shadow-[0_8px_32px_0_rgba(0,0,0,0.5)]">
                      <CardHeader>
                        <CardTitle className="text-xl">Analyze Post</CardTitle>
                        <CardDescription className="text-neutral-400">
                          Upload a screenshot of a social media post (containing text and/or images) for a comprehensive analysis.
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        <div className="flex flex-col gap-4">
                          <Label htmlFor="post-image-upload" className="cursor-pointer group">
                            <div className={`border-2 border-dashed rounded-xl flex flex-col items-center justify-center p-8 transition-all ${postImagePreview ? 'border-neutral-700 bg-black/30' : 'border-neutral-700 hover:border-neutral-500 bg-neutral-950/50 hover:bg-neutral-900/50'}`}>
                              {postImagePreview ? (
                                <div className="relative w-full aspect-video rounded-lg overflow-hidden flex items-center justify-center bg-black/50">
                                  <img src={postImagePreview} alt="Preview" className="max-h-full object-contain" />
                                </div>
                              ) : (
                                <div className="flex flex-col items-center gap-3 text-neutral-400 group-hover:text-neutral-300">
                                  <UploadCloud className="w-10 h-10" />
                                  <span className="font-medium">Click to upload post screenshot</span>
                                  <span className="text-xs text-neutral-500">PNG, JPG or WEBP (max. 5MB)</span>
                                </div>
                              )}
                            </div>
                            <Input 
                              id="post-image-upload" 
                              type="file" 
                              accept="image/*" 
                              className="hidden" 
                              onChange={handlePostImageUpload}
                            />
                          </Label>
                        </div>
                      </CardContent>
                      <CardFooter>
                        <Button 
                          onClick={handlePostAnalyze} 
                          disabled={!postImageFile || isPostLoading}
                          className="w-full bg-white text-black hover:bg-neutral-200 hover:scale-[1.02] active:scale-[0.98] shadow-[0_0_20px_rgba(255,255,255,0.1)] transition-all duration-300 font-semibold rounded-lg py-6"
                        >
                          {isPostLoading ? "Analyzing Post..." : "Analyze Post"}
                        </Button>
                      </CardFooter>
                    </Card>
                  </motion.div>
                )}

                {activeTab === "video" && (
                  <motion.div
                    key="video-input"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    transition={{ duration: 0.3 }}
                  >
                    <Card className="bg-slate-900/40 border-white/10 border-t-white/20 backdrop-blur-2xl shadow-[0_8px_32px_0_rgba(0,0,0,0.5)]">
                      <CardHeader>
                        <CardTitle className="text-xl">Detect Deepfake Video</CardTitle>
                        <CardDescription className="text-neutral-400">
                          Upload a video to check for AI manipulation or deepfakes (max 12MB).
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        <div className="flex flex-col gap-4">
                          <Label htmlFor="video-upload" className="cursor-pointer group">
                            <div className={`border-2 border-dashed rounded-xl flex flex-col items-center justify-center p-8 transition-all ${videoPreview ? 'border-neutral-700 bg-black/30' : 'border-neutral-700 hover:border-neutral-500 bg-neutral-950/50 hover:bg-neutral-900/50'}`}>
                              {videoPreview ? (
                                <div className="relative w-full aspect-video rounded-lg overflow-hidden flex items-center justify-center bg-black/50">
                                  <video src={videoPreview} controls className="max-h-full object-contain" />
                                </div>
                              ) : (
                                <div className="flex flex-col items-center gap-3 text-neutral-400 group-hover:text-neutral-300">
                                  <UploadCloud className="w-10 h-10" />
                                  <span className="font-medium">Click to upload or drag and drop</span>
                                  <span className="text-xs text-neutral-500">MP4, WEBM or MOV (max. 12MB)</span>
                                </div>
                              )}
                            </div>
                            <Input 
                              id="video-upload" 
                              type="file" 
                              accept="video/*" 
                              className="hidden" 
                              onChange={handleVideoUpload}
                            />
                          </Label>
                        </div>
                      </CardContent>
                      <CardFooter>
                        <Button 
                          onClick={handleVideoAnalyze} 
                          disabled={!videoFile || isVideoLoading}
                          className="w-full bg-white text-black hover:bg-neutral-200 hover:scale-[1.02] active:scale-[0.98] shadow-[0_0_20px_rgba(255,255,255,0.1)] transition-all duration-300 font-semibold rounded-lg py-6"
                        >
                          {isVideoLoading ? "Analyzing Video..." : "Analyze Video"}
                        </Button>
                      </CardFooter>
                    </Card>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* Results Column */}
            <div className="flex flex-col lg:col-span-7">
              <AnimatePresence mode="wait">
                {activeTab === "text" && textResult && !isTextLoading && (
                  <motion.div
                    key="text-result"
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.4, type: "spring" }}
                  >
                    <Card className="bg-neutral-900/60 border-white/10 backdrop-blur-xl shadow-2xl overflow-hidden relative">
                      <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-red-500 to-orange-500" />
                      <CardHeader className="pb-4">
                        <div className="flex justify-between items-start">
                          <div>
                            <CardTitle className="text-xl flex items-center gap-2">
                              <ShieldAlert className="w-5 h-5 text-neutral-400" />
                              Detection Result
                            </CardTitle>
                          </div>
                          {renderVerdictBadge(textResult.verdict)}
                        </div>
                      </CardHeader>
                      <CardContent className="flex flex-col gap-6">
                        <div className="space-y-2">
                          <div className="flex justify-between items-center text-sm">
                            <span className="text-neutral-400">
                              {textResult.verdict === "Verified" ? "Verification Confidence" : "Confidence Score"}
                            </span>
                            <span className="font-mono font-bold text-white">{Math.round(textResult.confidence * 100)}%</span>
                          </div>
                          <Progress 
                            value={textResult.confidence * 100} 
                            className="h-2 bg-neutral-800" 
                            indicatorClassName={textResult.verdict === "Verified" ? "bg-green-500" : "bg-red-500"} 
                          />
                        </div>
                        
                        <div className="space-y-2 bg-black/40 p-4 rounded-xl border border-white/5">
                          <h4 className="text-sm font-semibold text-neutral-300">Explanation</h4>
                          <p className="text-sm text-neutral-400 leading-relaxed">
                            {textResult.explanation}
                          </p>
                        </div>

                        {textResult.sources && textResult.sources.length > 0 && (
                          <div className="space-y-3">
                            <h4 className="text-sm font-semibold text-neutral-300">Supporting Sources</h4>
                            <div className="flex flex-col gap-2">
                              {textResult.sources.map((source: any, idx: number) => (
                                <a key={idx} href={source.url !== "#" ? source.url : undefined} target="_blank" rel="noreferrer" className="text-sm text-blue-400 hover:text-blue-300 hover:underline flex items-center gap-2 p-2 rounded-lg hover:bg-white/5 transition-colors">
                                  <FileText className="w-4 h-4" />
                                  {source.title || source.domain || "Reference Source"}
                                </a>
                              ))}
                            </div>
                          </div>
                        )}
                      </CardContent>
                    </Card>
                  </motion.div>
                )}

                {activeTab === "image" && imageResult && !isImageLoading && (
                  <motion.div
                    key="image-result"
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.4, type: "spring" }}
                  >
                    <Card className="bg-neutral-900/60 border-white/10 backdrop-blur-xl shadow-2xl overflow-hidden relative">
                      <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-red-500 to-purple-500" />
                      <CardHeader className="pb-4">
                        <div className="flex justify-between items-start">
                          <div>
                            <CardTitle className="text-xl flex items-center gap-2">
                              <ImageIcon className="w-5 h-5 text-neutral-400" />
                              {imageAnalysisMode === "seg_only" ? "Anomaly Detection Result" : "Analysis Result"}
                            </CardTitle>
                          </div>
                          {imageAnalysisMode !== "seg_only" && renderVerdictBadge(imageResult.verdict)}
                        </div>
                      </CardHeader>
                      <CardContent className="flex flex-col gap-6">
                        {imageAnalysisMode !== "seg_only" && (
                          <div className="space-y-2">
                            <div className="flex justify-between items-center text-sm">
                              <span className="text-neutral-400">
                                {imageResult.verdict === "AI Generated" ? "AI Probability" : "Authenticity Score"}
                              </span>
                              <span className="font-mono font-bold text-white">{Math.round(imageResult.confidence * 100)}%</span>
                            </div>
                            <Progress 
                              value={imageResult.confidence * 100} 
                              className="h-2 bg-neutral-800" 
                              indicatorClassName={imageResult.verdict === "AI Generated" ? "bg-purple-500" : "bg-green-500"} 
                            />
                          </div>
                        )}

                        {/* View selector switch if segmentation is available and not in seg_only mode */}
                        {imageResult.segmentationUrl && imageAnalysisMode !== "seg_only" && (
                          <div className="flex justify-between items-center bg-black/40 p-1 rounded-xl border border-white/5">
                            <button
                              type="button"
                              onClick={() => setImageResultView("classification")}
                              className={`flex-1 py-1.5 px-3 rounded-lg text-xs font-semibold transition-all ${imageResultView === "classification" ? "bg-neutral-800 text-white shadow-md" : "text-neutral-500 hover:text-neutral-300"}`}
                            >
                              Classification Report
                            </button>
                            <button
                              type="button"
                              onClick={() => setImageResultView("segmentation")}
                              className={`flex-1 py-1.5 px-3 rounded-lg text-xs font-semibold transition-all ${imageResultView === "segmentation" ? "bg-neutral-800 text-white shadow-md" : "text-neutral-500 hover:text-neutral-300"}`}
                            >
                              Segmentation Heatmap
                            </button>
                          </div>
                        )}
                        
                        <div className="space-y-3">
                          <div className="flex justify-between items-center">
                            <h4 className="text-sm font-semibold text-neutral-300">
                              {(imageAnalysisMode === "seg_only" || (imageResultView === "segmentation" && imageResult.segmentationUrl)) ? "Anomaly Heatmap" : "Forensic Report"}
                            </h4>
                            <span className="text-[11px] text-neutral-500 font-mono">Click image to enlarge</span>
                          </div>
                          <div 
                            className="relative w-full h-[650px] rounded-xl overflow-hidden bg-black/80 border border-white/10 flex items-center justify-center cursor-zoom-in group transition-all duration-300 hover:border-white/20"
                            onClick={() => setSelectedReportUrl((imageAnalysisMode === "seg_only" || imageResultView === "segmentation") && imageResult.segmentationUrl ? imageResult.segmentationUrl : imageResult.classificationUrl)}
                          >
                             <img 
                               src={(imageAnalysisMode === "seg_only" || imageResultView === "segmentation") && imageResult.segmentationUrl ? imageResult.segmentationUrl : imageResult.classificationUrl} 
                               className="max-h-full max-w-full object-contain transition-transform duration-300 group-hover:scale-[1.02]" 
                               alt="Heatmap/Report" 
                             />
                             {apiStatus === "demo" ? (
                               <div className="absolute inset-0 flex items-center justify-center bg-black/40 backdrop-blur-xs pointer-events-none">
                                  <span className="text-xs font-mono bg-black/80 px-3 py-1.5 rounded-full text-white border border-white/10">
                                    {(imageAnalysisMode === "seg_only" || imageResultView === "segmentation") && imageResult.segmentationUrl ? "Heatmap view (Click to zoom)" : "Report card view (Click to zoom)"}
                                  </span>
                                </div>
                             ) : (
                               <div className="absolute bottom-4 right-4 opacity-0 group-hover:opacity-100 transition-opacity bg-black/75 px-2.5 py-1 rounded-md text-[10px] font-mono text-neutral-400 border border-white/10 pointer-events-none">
                                 Click to zoom
                               </div>
                             )}
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </motion.div>
                )}

                {activeTab === "post" && postResult && !isPostLoading && (
                  <motion.div
                    key="post-result"
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.4, type: "spring" }}
                  >
                    <Card className="bg-neutral-900/60 border-white/10 backdrop-blur-xl shadow-2xl overflow-hidden relative">
                      <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-orange-500 to-yellow-500" />
                      <CardHeader className="pb-4">
                        <div className="flex justify-between items-start">
                          <div>
                            <CardTitle className="text-xl flex items-center gap-2">
                              <Layout className="w-5 h-5 text-neutral-400" />
                              Post Analysis Result
                            </CardTitle>
                          </div>
                          {renderVerdictBadge(postResult.verdict)}
                        </div>
                      </CardHeader>
                      <CardContent className="flex flex-col gap-6 max-h-[600px] overflow-y-auto pr-2 glass-scrollbar">
                        {/* Sub-view selector switch */}
                        <div className="flex justify-between items-center bg-black/40 p-1 rounded-xl border border-white/5">
                          <button
                            type="button"
                            onClick={() => setPostResultView("overall")}
                            className={`flex-1 py-1.5 px-3 rounded-lg text-xs font-semibold transition-all ${postResultView === "overall" ? "bg-neutral-800 text-white shadow-md" : "text-neutral-500 hover:text-neutral-300"}`}
                          >
                            Overall Status
                          </button>
                          <button
                            type="button"
                            onClick={() => setPostResultView("text")}
                            className={`flex-1 py-1.5 px-3 rounded-lg text-xs font-semibold transition-all ${postResultView === "text" ? "bg-neutral-800 text-white shadow-md" : "text-neutral-500 hover:text-neutral-300"}`}
                          >
                            Text Claims
                          </button>
                          <button
                            type="button"
                            onClick={() => setPostResultView("image")}
                            className={`flex-1 py-1.5 px-3 rounded-lg text-xs font-semibold transition-all ${postResultView === "image" ? "bg-neutral-800 text-white shadow-md" : "text-neutral-500 hover:text-neutral-300"}`}
                          >
                            Image Forensics
                          </button>
                        </div>

                        <div className="space-y-4">
                          {/* PAGE 1: OVERALL ANALYSIS */}
                          {postResultView === "overall" && (
                            <motion.div 
                              initial={{ opacity: 0, y: 5 }} 
                              animate={{ opacity: 1, y: 0 }} 
                              className="space-y-4"
                            >
                              <div className="space-y-2">
                                <div className="flex justify-between items-center text-sm">
                                  <span className="text-neutral-400">Overall Reliability Rating</span>
                                  <span className="font-mono font-bold text-white">{Math.round((1 - postResult.confidence) * 100)}%</span>
                                </div>
                                <Progress value={(1 - postResult.confidence) * 100} className="h-2 bg-neutral-800" indicatorClassName="bg-yellow-500" />
                              </div>

                              <div className="space-y-2 bg-black/40 p-4 rounded-xl border border-white/5">
                                <h4 className="text-sm font-semibold text-neutral-300">Overall Explanation</h4>
                                <p className="text-sm text-neutral-400 leading-relaxed">
                                  {postResult.explanation}
                                </p>
                              </div>
                            </motion.div>
                          )}

                          {/* PAGE 2: TEXT CLAIMS */}
                          {postResultView === "text" && (
                            <motion.div 
                              initial={{ opacity: 0, y: 5 }} 
                              animate={{ opacity: 1, y: 0 }}
                              className="space-y-3"
                            >
                              <h4 className="text-xs font-semibold text-neutral-400 uppercase tracking-wider">Extracted Claims ({postResult.textResults?.length || 0})</h4>
                              {postResult.textResults && postResult.textResults.length > 0 ? (
                                <div className="space-y-3">
                                  {postResult.textResults.map((tr: any, idx: number) => (
                                    <div key={idx} className="bg-black/40 border border-white/5 rounded-xl p-4 space-y-4">
                                      <div className="flex justify-between items-center border-b border-white/5 pb-2">
                                        <span className="text-xs font-mono text-neutral-500 capitalize">{tr.source} context</span>
                                        {renderVerdictBadge(tr.verdict)}
                                      </div>
                                      <p className="text-xs text-neutral-400 italic bg-black/20 p-2 rounded-lg border border-white/5">
                                        "{tr.raw.slice(0, 150)}{tr.raw.length > 150 ? '...' : ''}"
                                      </p>

                                      {tr.verdict !== "No Claim" && (
                                        <div className="space-y-2">
                                          <div className="flex justify-between items-center text-xs">
                                            <span className="text-neutral-400">
                                              {tr.verdict === "Verified" ? "Verification Confidence" : "Confidence Score"}
                                            </span>
                                            <span className="font-mono font-bold text-white">{Math.round((tr.confidence || 0.8) * 100)}%</span>
                                          </div>
                                          <Progress 
                                            value={(tr.confidence || 0.8) * 100} 
                                            className="h-1.5 bg-neutral-800" 
                                            indicatorClassName={tr.verdict === "Verified" ? "bg-green-500" : "bg-red-500"} 
                                          />
                                        </div>
                                      )}

                                      <p className="text-sm text-neutral-300 leading-relaxed">{tr.explanation}</p>
                                      
                                      {tr.sources && tr.sources.length > 0 && (
                                        <div className="flex flex-col gap-1.5 pt-2 border-t border-white/5">
                                          <span className="text-[10px] text-neutral-500 font-semibold uppercase tracking-wider">References:</span>
                                          {tr.sources.slice(0, 2).map((s: any, sIdx: number) => (
                                            <a key={sIdx} href={s.url !== "#" ? s.url : undefined} target="_blank" rel="noreferrer" className="text-xs text-blue-400 hover:text-blue-300 hover:underline flex items-center gap-1">
                                              • {s.title || s.domain}
                                            </a>
                                          ))}
                                        </div>
                                      )}
                                    </div>
                                  ))}
                                </div>
                              ) : (
                                <p className="text-neutral-500 text-sm text-center py-8">No text claims were extracted from this post.</p>
                              )}
                            </motion.div>
                          )}

                          {/* PAGE 3: IMAGE FORENSICS */}
                          {postResultView === "image" && (
                            <motion.div 
                              initial={{ opacity: 0, y: 5 }} 
                              animate={{ opacity: 1, y: 0 }}
                              className="space-y-3"
                            >
                              <h4 className="text-xs font-semibold text-neutral-400 uppercase tracking-wider">Detected Image Regions ({postResult.regions?.length || 0})</h4>
                              {postResult.regions && postResult.regions.length > 0 ? (
                                <div className="grid grid-cols-1 gap-3">
                                  {postResult.regions.map((reg: any) => {
                                    const isFake = !reg.label.toLowerCase().includes("real") && !reg.label.toLowerCase().includes("auth");
                                    const labelText = isFake ? "AI Generated" : "Authentic Image";
                                    return (
                                      <div key={reg.id} className="bg-black/30 border border-white/5 rounded-xl p-4 flex flex-col gap-4 hover:bg-white/5 transition-colors">
                                        <div className="flex justify-between items-center">
                                          <div className="flex flex-col">
                                            <span className="text-sm font-semibold text-white capitalize">{reg.type}</span>
                                          </div>
                                          <div className="flex items-center gap-3">
                                            {renderVerdictBadge(labelText)}
                                            {reg.classificationUrl && (
                                              <button
                                                onClick={() => setSelectedReportUrl(reg.classificationUrl)}
                                                className="text-xs text-blue-400 hover:text-blue-300 hover:underline bg-transparent border-0 cursor-zoom-in"
                                              >
                                                Report Card
                                              </button>
                                            )}
                                          </div>
                                        </div>

                                        <div className="space-y-2">
                                          <div className="flex justify-between items-center text-xs">
                                            <span className="text-neutral-400">
                                              {isFake ? "AI Probability" : "Authenticity Score"}
                                            </span>
                                            <span className="font-mono font-bold text-white">{Math.round(reg.confidence * 100)}%</span>
                                          </div>
                                          <Progress 
                                            value={reg.confidence * 100} 
                                            className="h-1.5 bg-neutral-800" 
                                            indicatorClassName={isFake ? "bg-purple-500" : "bg-green-500"} 
                                          />
                                        </div>
                                        
                                        {reg.segmentationUrl && (
                                          <div className="space-y-2">
                                            <span className="text-xs font-semibold text-red-400/90 uppercase tracking-wider block">Localization Heatmap (Pixel Anomaly)</span>
                                            <div 
                                              className="relative w-full h-[280px] rounded-lg overflow-hidden bg-black/60 border border-red-500/10 flex items-center justify-center cursor-zoom-in group"
                                              onClick={() => setSelectedReportUrl(reg.segmentationUrl)}
                                            >
                                              <img 
                                                src={reg.segmentationUrl} 
                                                className="max-h-full max-w-full object-contain transition-transform duration-300 group-hover:scale-[1.02]" 
                                                alt="Sub-region Heatmap" 
                                              />
                                              <div className="absolute bottom-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity bg-black/75 px-2.5 py-1 rounded text-[10px] font-mono text-neutral-400 border border-white/10 pointer-events-none">
                                                Click to zoom
                                              </div>
                                            </div>
                                          </div>
                                        )}
                                      </div>
                                    );
                                  })}
                                </div>
                              ) : (
                                <p className="text-neutral-500 text-sm text-center py-8">No image regions were extracted from this post.</p>
                              )}
                            </motion.div>
                          )}
                        </div>
                      </CardContent>
                    </Card>
                  </motion.div>
                )}

                {activeTab === "video" && videoResult && !isVideoLoading && (
                  <motion.div
                    key="video-result"
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.4, type: "spring" }}
                  >
                    <Card className="bg-neutral-900/60 border-white/10 backdrop-blur-xl shadow-2xl overflow-hidden relative">
                      <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-red-500 to-blue-500" />
                      <CardHeader className="pb-4">
                        <div className="flex justify-between items-start">
                          <div>
                            <CardTitle className="text-xl flex items-center gap-2">
                              <Video className="w-5 h-5 text-neutral-400" />
                              Video Analysis Result
                            </CardTitle>
                          </div>
                          {renderVerdictBadge(videoResult.verdict)}
                        </div>
                      </CardHeader>
                      <CardContent className="flex flex-col gap-6">
                        <div className="space-y-2">
                          <div className="flex justify-between items-center text-sm">
                            <span className="text-neutral-400">
                              {videoResult.verdict === "Deepfake Detected" ? "Deepfake Probability" : "Authenticity Confidence"}
                            </span>
                            <span className="font-mono font-bold text-white">{Math.round(videoResult.confidence * 100)}%</span>
                          </div>
                          <Progress 
                            value={videoResult.confidence * 100} 
                            className="h-2 bg-neutral-800" 
                            indicatorClassName={videoResult.verdict === "Deepfake Detected" ? "bg-red-500" : "bg-green-500"} 
                          />
                        </div>
                        
                        <div className="space-y-2 bg-black/40 p-4 rounded-xl border border-white/5">
                          <h4 className="text-sm font-semibold text-neutral-300">Explanation</h4>
                          <p className="text-sm text-neutral-400 leading-relaxed">
                            {videoResult.explanation}
                          </p>
                        </div>
                      </CardContent>
                    </Card>
                  </motion.div>
                )}

                {/* Loading State Placeholder */}
                {isLoading && (
                  <motion.div
                    key="loading"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="flex flex-col items-center justify-center h-full min-h-[300px] border border-white/5 rounded-xl bg-neutral-900/20 backdrop-blur-sm"
                  >
                    <div className="w-12 h-12 border-4 border-neutral-700 border-t-white rounded-full animate-spin mb-4" />
                    <p className="text-neutral-400 animate-pulse">
                      {isTextLoading && "Analyzing text against databases..."}
                      {isImageLoading && "Scanning image artifacts..."}
                      {isPostLoading && "Cross-referencing post content..."}
                      {isVideoLoading && "Analyzing video frames..."}
                    </p>
                  </motion.div>
                )}
                
                {/* Empty State */}
                {((activeTab === "text" && !textResult && !isTextLoading) || 
                  (activeTab === "image" && !imageResult && !isImageLoading) ||
                  (activeTab === "post" && !postResult && !isPostLoading) ||
                  (activeTab === "video" && !videoResult && !isVideoLoading)) && (
                  <motion.div
                    key="empty"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="flex flex-col items-center justify-center h-full min-h-[300px] border border-white/5 border-dashed rounded-xl bg-neutral-900/10"
                  >
                    <div className="w-16 h-16 rounded-full bg-neutral-900 flex items-center justify-center mb-4 border border-white/5 shadow-inner overflow-hidden">
                      <img src="/logo.png" alt="RealEyes Logo" className="w-10 h-10 object-contain opacity-40 grayscale" />
                    </div>
                    <h3 className="text-lg font-medium text-neutral-300 mb-1">Awaiting Input</h3>
                    <p className="text-neutral-500 text-sm text-center max-w-[250px]">
                      Provide {activeTab === "text" ? "text" : activeTab === "image" ? "an image" : activeTab === "post" ? "a post" : "a video"} to receive a detailed authenticity analysis.
                    </p>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </Tabs>
      </main>

      <AnimatePresence>
        {selectedReportUrl && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-black/90 backdrop-blur-md p-4 md:p-8 cursor-zoom-out"
            onClick={() => setSelectedReportUrl(null)}
          >
            <div className="relative max-w-5xl max-h-[85vh] w-full flex items-center justify-center">
              <motion.div
                initial={{ scale: 0.95, y: 15 }}
                animate={{ scale: 1, y: 0 }}
                exit={{ scale: 0.95, y: 15 }}
                transition={{ type: "spring", damping: 25, stiffness: 200 }}
                className="relative bg-neutral-950/80 border border-white/10 rounded-2xl overflow-hidden p-2 shadow-2xl max-w-full max-h-full flex flex-col"
                onClick={(e) => e.stopPropagation()}
              >
                <button
                  onClick={() => setSelectedReportUrl(null)}
                  className="absolute top-4 right-4 bg-black/70 hover:bg-black/90 text-white rounded-full p-2.5 border border-white/15 transition-all z-20 hover:scale-105"
                  aria-label="Close modal"
                >
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
                <div className="overflow-auto flex items-center justify-center p-2">
                  <img
                    src={selectedReportUrl}
                    alt="Enlarged Forensic Visualization"
                    className="max-w-full max-h-[75vh] object-contain rounded-xl shadow-inner border border-white/5"
                  />
                </div>
                <div className="px-6 py-4 border-t border-white/5 bg-neutral-900/40 text-center">
                  <h4 className="text-sm font-semibold text-neutral-200">High-Resolution Forensic Analysis Report</h4>
                  <p className="text-xs text-neutral-400 mt-1">Detailed pixel-level classification and anomaly localization map</p>
                </div>
              </motion.div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
