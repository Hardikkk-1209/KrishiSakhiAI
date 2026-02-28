"use client";

import { useEffect, useRef, useCallback, useTransition } from "react";
import { useState } from "react";
import { cn } from "@/lib/utils";
import {
    ImageIcon,
    FileUp,
    Figma,
    MonitorIcon,
    CircleUserRound,
    ArrowUpIcon,
    Paperclip,
    PlusIcon,
    SendIcon,
    XIcon,
    LoaderIcon,
    Sparkles,
    Command,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import * as React from "react"

interface UseAutoResizeTextareaProps {
    minHeight: number;
    maxHeight?: number;
}

function useAutoResizeTextarea({
    minHeight,
    maxHeight,
}: UseAutoResizeTextareaProps) {
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    const adjustHeight = useCallback(
        (reset?: boolean) => {
            const textarea = textareaRef.current;
            if (!textarea) return;

            if (reset) {
                textarea.style.height = `${minHeight}px`;
                return;
            }

            textarea.style.height = `${minHeight}px`;
            const newHeight = Math.max(
                minHeight,
                Math.min(
                    textarea.scrollHeight,
                    maxHeight ?? Number.POSITIVE_INFINITY
                )
            );

            textarea.style.height = `${newHeight}px`;
        },
        [minHeight, maxHeight]
    );

    useEffect(() => {
        const textarea = textareaRef.current;
        if (textarea) {
            textarea.style.height = `${minHeight}px`;
        }
    }, [minHeight]);

    useEffect(() => {
        const handleResize = () => adjustHeight();
        window.addEventListener("resize", handleResize);
        return () => window.removeEventListener("resize", handleResize);
    }, [adjustHeight]);

    return { textareaRef, adjustHeight };
}

interface CommandSuggestion {
    icon: React.ReactNode;
    label: string;
    description: string;
    prefix: string;
}

interface TextareaProps
    extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
    containerClassName?: string;
    showRing?: boolean;
}

const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(
    ({ className, containerClassName, showRing = true, ...props }, ref) => {
        const [isFocused, setIsFocused] = React.useState(false);

        return (
            <div className={cn(
                "relative",
                containerClassName
            )}>
                <textarea
                    className={cn(
                        "flex min-h-[80px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm",
                        "transition-all duration-200 ease-in-out",
                        "placeholder:text-muted-foreground",
                        "disabled:cursor-not-allowed disabled:opacity-50",
                        showRing ? "focus-visible:outline-none focus-visible:ring-0 focus-visible:ring-offset-0" : "",
                        className
                    )}
                    ref={ref}
                    onFocus={() => setIsFocused(true)}
                    onBlur={() => setIsFocused(false)}
                    {...props}
                />

                {showRing && isFocused && (
                    <motion.span
                        className="absolute inset-0 rounded-md pointer-events-none ring-2 ring-offset-0 ring-violet-500/30"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        transition={{ duration: 0.2 }}
                    />
                )}

                {props.onChange && (
                    <div
                        className="absolute bottom-2 right-2 opacity-0 w-2 h-2 bg-violet-500 rounded-full"
                        style={{
                            animation: 'none',
                        }}
                        id="textarea-ripple"
                    />
                )}
            </div>
        )
    }
)
Textarea.displayName = "Textarea"

// ═══════════════════════════════════════════
// KRISHI SAKHI CUSTOMIZED CHAT
// ═══════════════════════════════════════════

interface ChatMessage {
    role: 'user' | 'assistant';
    content: string;
    image?: string;
}

interface AnimatedAIChatProps {
    language: string;
    model: string;
    temperature: number;
    farmerProfile: Record<string, string> | null;
}

export function AnimatedAIChat({ language, model, temperature, farmerProfile }: AnimatedAIChatProps) {
    const [value, setValue] = useState("");
    const [attachments, setAttachments] = useState<string[]>([]);
    const [isTyping, setIsTyping] = useState(false);
    const [isPending, startTransition] = useTransition();
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [imageFile, setImageFile] = useState<File | null>(null);
    const [imagePreview, setImagePreview] = useState<string | null>(null);
    const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
    const { textareaRef, adjustHeight } = useAutoResizeTextarea({
        minHeight: 60,
        maxHeight: 200,
    });
    const [inputFocused, setInputFocused] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const farmCommands: CommandSuggestion[] = [
        { icon: <ImageIcon className="w-4 h-4" />, label: "Scan Crop", description: "Upload a crop photo for disease analysis", prefix: "/scan" },
        { icon: <Sparkles className="w-4 h-4" />, label: "Crop Advice", description: "Get seasonal crop recommendations", prefix: "/crop" },
        { icon: <MonitorIcon className="w-4 h-4" />, label: "Market Rates", description: "Check current market rates", prefix: "/market" },
        { icon: <Figma className="w-4 h-4" />, label: "Soil Tips", description: "Get soil health improvement tips", prefix: "/soil" },
    ];

    const [showCommandPalette, setShowCommandPalette] = useState(false);
    const [activeSuggestion, setActiveSuggestion] = useState(-1);
    const commandPaletteRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (value.startsWith('/') && !value.includes(' ')) {
            setShowCommandPalette(true);
            const idx = farmCommands.findIndex(c => c.prefix.startsWith(value));
            setActiveSuggestion(idx >= 0 ? idx : -1);
        } else {
            setShowCommandPalette(false);
        }
    }, [value]);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, isTyping]);

    useEffect(() => {
        const handleMouseMove = (e: MouseEvent) => setMousePosition({ x: e.clientX, y: e.clientY });
        window.addEventListener('mousemove', handleMouseMove);
        return () => window.removeEventListener('mousemove', handleMouseMove);
    }, []);

    const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;
        setImageFile(file);
        const reader = new FileReader();
        reader.onload = (ev) => setImagePreview(ev.target?.result as string);
        reader.readAsDataURL(file);
    };

    const clearImage = () => {
        setImageFile(null);
        setImagePreview(null);
        if (fileInputRef.current) fileInputRef.current.value = '';
    };

    const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
        if (showCommandPalette) {
            if (e.key === 'ArrowDown') { e.preventDefault(); setActiveSuggestion(p => p < farmCommands.length - 1 ? p + 1 : 0); }
            else if (e.key === 'ArrowUp') { e.preventDefault(); setActiveSuggestion(p => p > 0 ? p - 1 : farmCommands.length - 1); }
            else if (e.key === 'Tab' || e.key === 'Enter') {
                e.preventDefault();
                if (activeSuggestion >= 0) { selectCommand(activeSuggestion); }
            } else if (e.key === 'Escape') { e.preventDefault(); setShowCommandPalette(false); }
        } else if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            if (value.trim() || imageFile) handleSendMessage();
        }
    };

    const selectCommand = (index: number) => {
        const cmd = farmCommands[index];
        if (cmd.prefix === '/scan') {
            fileInputRef.current?.click();
        }
        setValue(cmd.prefix + ' ');
        setShowCommandPalette(false);
    };

    const handleSendMessage = async () => {
        const msg = value.trim();
        if (!msg && !imageFile) return;

        const userMsg: ChatMessage = { role: 'user', content: msg || 'Analyze this image' };
        if (imagePreview) userMsg.image = imagePreview;
        setMessages(prev => [...prev, userMsg]);
        setValue("");
        adjustHeight(true);
        setIsTyping(true);

        try {
            if (imageFile) {
                // Image analysis
                const formData = new FormData();
                formData.append('file', imageFile);
                formData.append('question', msg || 'What crop disease or pest do you see? Provide diagnosis and treatment.');
                formData.append('language', language);
                formData.append('model', 'llava');

                const resp = await fetch('/api/analyze-image', { method: 'POST', body: formData });
                const data = await resp.json();
                setMessages(prev => [...prev, { role: 'assistant', content: data.analysis }]);
                clearImage();
            } else {
                // Text chat
                const resp = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        message: msg,
                        model: model,
                        temperature: temperature,
                        history: messages.slice(-10).map(m => ({ role: m.role, content: m.content })),
                        farmer_profile: farmerProfile,
                        language: language,
                    }),
                });

                const reader = resp.body?.getReader();
                const decoder = new TextDecoder();
                let assistantContent = '';

                setMessages(prev => [...prev, { role: 'assistant', content: '' }]);

                if (reader) {
                    while (true) {
                        const { done, value: chunk } = await reader.read();
                        if (done) break;
                        const text = decoder.decode(chunk);
                        for (const line of text.split('\n')) {
                            if (line.startsWith('data: ')) {
                                const token = line.slice(6);
                                if (token === '[DONE]') break;
                                assistantContent += token;
                                setMessages(prev => {
                                    const updated = [...prev];
                                    updated[updated.length - 1] = { role: 'assistant', content: assistantContent };
                                    return updated;
                                });
                            }
                        }
                    }
                }
            }
        } catch (err) {
            setMessages(prev => [...prev, { role: 'assistant', content: '❌ Failed to connect to AI. Make sure the server is running on port 8000.' }]);
        }
        setIsTyping(false);
    };

    return (
        <div className="flex-1 flex flex-col h-full bg-transparent text-white relative overflow-hidden">
            {/* Background glows */}
            <div className="absolute inset-0 w-full h-full overflow-hidden pointer-events-none">
                <div className="absolute top-0 left-1/4 w-96 h-96 bg-emerald-500/5 rounded-full mix-blend-normal filter blur-[128px] animate-pulse" />
                <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-green-500/5 rounded-full mix-blend-normal filter blur-[128px] animate-pulse delay-700" />
            </div>

            {/* Messages Area */}
            <div className="flex-1 overflow-y-auto px-6 py-4 relative z-10">
                {messages.length === 0 && (
                    <motion.div
                        className="w-full max-w-2xl mx-auto mt-16 text-center space-y-6 flex flex-col items-center"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                    >
                        <h1 className="text-3xl font-medium tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-emerald-300 to-green-500 pb-1 text-center w-full">
                            🙏 Namaste{farmerProfile?.name ? `, ${farmerProfile.name}` : ''}!
                        </h1>
                        <p className="text-sm text-white/40 text-center w-full">
                            Ask me about crops, pests, soil health, market trends, or upload a photo for disease analysis
                        </p>
                        <div className="flex flex-wrap items-center justify-center gap-2 mt-8">
                            {farmCommands.map((cmd, i) => (
                                <motion.button
                                    key={cmd.prefix}
                                    onClick={() => selectCommand(i)}
                                    className="flex items-center gap-2 px-3 py-2 bg-white/[0.02] hover:bg-white/[0.05] rounded-lg text-sm text-white/60 hover:text-white/90 transition-all border border-white/[0.05]"
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: i * 0.1 }}
                                >
                                    {cmd.icon}
                                    <span>{cmd.label}</span>
                                </motion.button>
                            ))}
                        </div>
                    </motion.div>
                )}

                {messages.map((msg, i) => (
                    <motion.div
                        key={i}
                        className={cn("flex gap-3 mb-4 max-w-3xl mx-auto", msg.role === 'user' ? 'flex-row-reverse' : '')}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                    >
                        <div className={cn(
                            "w-8 h-8 rounded-full flex items-center justify-center text-sm flex-shrink-0",
                            msg.role === 'user' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-white/5 text-white/60'
                        )}>
                            {msg.role === 'user' ? '👤' : '🌾'}
                        </div>
                        <div className={cn(
                            "max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed",
                            msg.role === 'user'
                                ? 'bg-emerald-500/10 border border-emerald-500/20 text-white/90'
                                : 'bg-white/[0.03] border border-white/[0.05] text-white/80'
                        )}>
                            {msg.image && (
                                <img src={msg.image} alt="Upload" className="max-w-[200px] rounded-lg mb-2" />
                            )}
                            <div className="whitespace-pre-wrap">{msg.content}</div>
                        </div>
                    </motion.div>
                ))}

                {isTyping && messages[messages.length - 1]?.role !== 'assistant' && (
                    <motion.div className="flex gap-3 mb-4 max-w-3xl mx-auto" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                        <div className="w-8 h-8 rounded-full bg-white/5 flex items-center justify-center text-sm">🌾</div>
                        <div className="bg-white/[0.03] border border-white/[0.05] rounded-2xl px-4 py-3 text-sm text-white/60">
                            <TypingDots />
                        </div>
                    </motion.div>
                )}
                <div ref={messagesEndRef} />
            </div>

            {/* Image Preview */}
            <AnimatePresence>
                {imagePreview && (
                    <motion.div className="px-6 pb-2" initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }}>
                        <div className="inline-flex items-center gap-2 bg-emerald-500/10 border border-emerald-500/20 rounded-lg px-3 py-2">
                            <img src={imagePreview} alt="Preview" className="w-12 h-12 object-cover rounded" />
                            <span className="text-xs text-emerald-400">📷 Image ready</span>
                            <button onClick={clearImage} className="text-white/40 hover:text-red-400"><XIcon className="w-3 h-3" /></button>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Input Area */}
            <div className="p-4 relative z-10">
                <div className="max-w-3xl mx-auto">
                    <AnimatePresence>
                        {showCommandPalette && (
                            <motion.div
                                ref={commandPaletteRef}
                                className="mb-2 backdrop-blur-xl bg-black/90 rounded-lg z-50 shadow-lg border border-white/10 overflow-hidden"
                                initial={{ opacity: 0, y: 5 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 5 }}
                            >
                                {farmCommands.map((cmd, i) => (
                                    <div
                                        key={cmd.prefix}
                                        className={cn("flex items-center gap-2 px-3 py-2 text-xs cursor-pointer transition-colors",
                                            activeSuggestion === i ? "bg-white/10 text-white" : "text-white/70 hover:bg-white/5"
                                        )}
                                        onClick={() => selectCommand(i)}
                                    >
                                        {cmd.icon}
                                        <span className="font-medium">{cmd.label}</span>
                                        <span className="text-white/40 ml-1">{cmd.description}</span>
                                    </div>
                                ))}
                            </motion.div>
                        )}
                    </AnimatePresence>

                    <motion.div
                        className="backdrop-blur-2xl bg-white/[0.02] rounded-2xl border border-white/[0.05] shadow-2xl"
                        initial={{ scale: 0.98 }} animate={{ scale: 1 }}
                    >
                        <div className="p-3">
                            <Textarea
                                ref={textareaRef}
                                value={value}
                                onChange={(e) => { setValue(e.target.value); adjustHeight(); }}
                                onKeyDown={handleKeyDown}
                                onFocus={() => setInputFocused(true)}
                                onBlur={() => setInputFocused(false)}
                                placeholder="Ask KrishiSakhi anything about farming..."
                                containerClassName="w-full"
                                className="w-full px-4 py-3 resize-none bg-transparent border-none text-white/90 text-sm focus:outline-none placeholder:text-white/20 min-h-[50px]"
                                style={{ overflow: "hidden" }}
                                showRing={false}
                            />
                        </div>
                        <div className="px-4 pb-3 flex items-center justify-between gap-3">
                            <div className="flex items-center gap-2">
                                <input type="file" ref={fileInputRef} accept="image/*" className="hidden" onChange={handleImageUpload} />
                                <motion.button
                                    onClick={() => fileInputRef.current?.click()}
                                    whileTap={{ scale: 0.94 }}
                                    className="p-2 text-white/40 hover:text-emerald-400 rounded-lg transition-colors"
                                    title="Upload crop/livestock photo"
                                >
                                    <ImageIcon className="w-4 h-4" />
                                </motion.button>
                                <motion.button
                                    onClick={() => setShowCommandPalette(p => !p)}
                                    whileTap={{ scale: 0.94 }}
                                    className={cn("p-2 text-white/40 hover:text-white/90 rounded-lg transition-colors", showCommandPalette && "bg-white/10 text-white/90")}
                                >
                                    <Command className="w-4 h-4" />
                                </motion.button>
                            </div>
                            <motion.button
                                onClick={handleSendMessage}
                                whileHover={{ scale: 1.01 }}
                                whileTap={{ scale: 0.98 }}
                                disabled={isTyping || (!value.trim() && !imageFile)}
                                className={cn(
                                    "px-4 py-2 rounded-lg text-sm font-medium transition-all flex items-center gap-2",
                                    (value.trim() || imageFile)
                                        ? "bg-emerald-500 text-white shadow-lg shadow-emerald-500/20"
                                        : "bg-white/[0.05] text-white/40"
                                )}
                            >
                                {isTyping ? <LoaderIcon className="w-4 h-4 animate-spin" /> : <SendIcon className="w-4 h-4" />}
                                <span>Send</span>
                            </motion.button>
                        </div>
                    </motion.div>
                </div>
            </div>

            {inputFocused && (
                <motion.div
                    className="fixed w-[50rem] h-[50rem] rounded-full pointer-events-none z-0 opacity-[0.02] bg-gradient-to-r from-emerald-500 via-green-500 to-teal-500 blur-[96px]"
                    animate={{ x: mousePosition.x - 400, y: mousePosition.y - 400 }}
                    transition={{ type: "spring", damping: 25, stiffness: 150, mass: 0.5 }}
                />
            )}
        </div>
    );
}

function TypingDots() {
    return (
        <div className="flex items-center">
            {[1, 2, 3].map((dot) => (
                <motion.div
                    key={dot}
                    className="w-1.5 h-1.5 bg-emerald-400 rounded-full mx-0.5"
                    animate={{ opacity: [0.3, 0.9, 0.3], scale: [0.85, 1.1, 0.85] }}
                    transition={{ duration: 1.2, repeat: Infinity, delay: dot * 0.15, ease: "easeInOut" }}
                />
            ))}
        </div>
    );
}
