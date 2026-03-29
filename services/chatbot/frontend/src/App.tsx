import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Bot, User, Send, Loader2, Settings, X } from 'lucide-react';

type Role = 'user' | 'assistant';

interface Message {
  role: Role;
  content: string;
}

const DEFAULT_SYSTEM = 'You are an assistant named Bill. Your job is to quote movies from the 90s, like Waynes World, Pulp Fiction, and The Big Lebowski. You should only respond with movie quotes, and never break character.';

export default function App() {
  const [displayMessages, setDisplayMessages] = useState<Message[]>([
    { role: 'assistant', content: 'Hello! How can I help you today?' },
  ]);
  // History sent to the LLM — excludes the synthetic welcome message
  const [history, setHistory] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [system, setSystem] = useState(DEFAULT_SYSTEM);
  const [showSettings, setShowSettings] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [displayMessages, loading]);

  const sendMessage = async () => {
    const text = input.trim();
    if (!text || loading) return;

    const userMessage: Message = { role: 'user', content: text };
    const nextHistory = [...history, userMessage];

    setDisplayMessages(prev => [...prev, userMessage]);
    setHistory(nextHistory);
    setInput('');
    setLoading(true);

    try {
      const res = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: nextHistory, system, max_tokens: 512 }),
      });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data = (await res.json()) as { response: string };
      const assistantMessage: Message = { role: 'assistant', content: data.response };
      setDisplayMessages(prev => [...prev, assistantMessage]);
      setHistory(prev => [...prev, assistantMessage]);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      setDisplayMessages(prev => [
        ...prev,
        { role: 'assistant', content: `**Error:** ${message}` },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      void sendMessage();
    }
  };

  return (
    <div className="flex flex-col h-full max-w-3xl mx-auto">
      {/* Header */}
      <header className="flex items-center gap-3 px-6 py-4 bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-md">
        <div className="flex items-center justify-center w-9 h-9 bg-white/20 rounded-full">
          <Bot className="w-5 h-5" />
        </div>
        <h1 className="text-lg font-semibold tracking-tight flex-1">AI Chatbot</h1>
        <button
          className="flex items-center justify-center w-8 h-8 rounded-full hover:bg-white/20 transition-colors"
          onClick={() => setShowSettings(s => !s)}
          aria-label="Settings"
        >
          {showSettings ? <X className="w-4 h-4" /> : <Settings className="w-4 h-4" />}
        </button>
      </header>

      {/* System prompt panel */}
      {showSettings && (
        <div className="bg-indigo-50 border-b border-indigo-200 px-6 py-4">
          <label className="block text-xs font-semibold text-indigo-700 mb-1 uppercase tracking-wide">
            System prompt
          </label>
          <textarea
            className="w-full resize-none rounded-lg border border-indigo-200 bg-white px-3 py-2 text-sm leading-relaxed outline-none focus:border-indigo-500 transition-colors"
            value={system}
            onChange={e => setSystem(e.target.value)}
            rows={3}
          />
        </div>
      )}

      {/* Messages */}
      <main className="flex-1 overflow-y-auto px-4 py-6 space-y-4 bg-slate-50">
        {displayMessages.map((msg, idx) => (
          <MessageRow key={idx} message={msg} />
        ))}
        {loading && <TypingIndicator />}
        <div ref={bottomRef} />
      </main>

      {/* Input */}
      <footer className="border-t border-slate-200 bg-white px-4 py-3">
        <div className="flex items-end gap-3">
          <textarea
            className="flex-1 resize-none rounded-xl border border-slate-300 bg-slate-50 px-4 py-2.5 text-sm leading-relaxed outline-none focus:border-blue-500 focus:bg-white transition-colors disabled:opacity-50"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type a message… (Enter to send, Shift+Enter for newline)"
            rows={2}
            disabled={loading}
          />
          <button
            className="flex items-center justify-center w-10 h-10 rounded-xl bg-blue-600 text-white hover:bg-blue-700 disabled:bg-slate-300 disabled:cursor-not-allowed transition-colors"
            onClick={() => void sendMessage()}
            disabled={loading || !input.trim()}
            aria-label="Send message"
          >
            {loading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Send className="w-4 h-4" />
            )}
          </button>
        </div>
      </footer>
    </div>
  );
}

function MessageRow({ message }: { message: Message }) {
  const isUser = message.role === 'user';
  return (
    <div className={`flex items-end gap-2 ${isUser ? 'flex-row-reverse' : ''}`}>
      <div
        className={`flex-shrink-0 flex items-center justify-center w-8 h-8 rounded-full text-white ${
          isUser ? 'bg-blue-500' : 'bg-indigo-500'
        }`}
      >
        {isUser ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
      </div>
      <div
        className={`max-w-[75%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
          isUser
            ? 'bg-blue-600 text-white rounded-br-sm'
            : 'bg-white text-slate-800 shadow-sm border border-slate-100 rounded-bl-sm'
        }`}
      >
        {isUser ? (
          <p className="whitespace-pre-wrap break-words">{message.content}</p>
        ) : (
          <div className="prose prose-sm max-w-none prose-p:my-1 prose-pre:bg-slate-100 prose-pre:text-slate-800 prose-code:text-blue-600 prose-code:bg-slate-100 prose-code:px-1 prose-code:rounded prose-headings:text-slate-900">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {message.content}
            </ReactMarkdown>
          </div>
        )}
      </div>
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className="flex items-end gap-2">
      <div className="flex-shrink-0 flex items-center justify-center w-8 h-8 rounded-full bg-indigo-500 text-white">
        <Bot className="w-4 h-4" />
      </div>
      <div className="bg-white shadow-sm border border-slate-100 rounded-2xl rounded-bl-sm px-4 py-3">
        <div className="flex gap-1 items-center">
          <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce [animation-delay:-0.3s]" />
          <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce [animation-delay:-0.15s]" />
          <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" />
        </div>
      </div>
    </div>
  );
}
