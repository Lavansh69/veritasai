import type { Metadata } from 'next';
import './globals.css';
import Navbar from '@/components/Navbar';
import { ThemeProvider } from '@/components/ThemeProvider';

export const metadata: Metadata = {
  title: 'VeritasAI — AI-Powered Deepfake Detection',
  description:
    'Upload suspicious images or videos to verify authenticity. Powered by advanced AI models, forensic metadata analysis, and explainable heatmaps.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body className="min-h-screen relative overflow-x-hidden">
        <ThemeProvider>
          {/* Background orbs */}
          <div className="fixed inset-0 pointer-events-none -z-10 overflow-hidden">
            <div className="glow-dot w-96 h-96 bg-brand-500 top-[-10%] left-[-5%]" />
            <div className="glow-dot w-80 h-80 bg-purple-500 top-[40%] right-[-8%]" />
            <div className="glow-dot w-72 h-72 bg-pink-500 bottom-[-5%] left-[30%]" />
          </div>

          <Navbar />
          <main className="pt-20">{children}</main>

          <footer className="text-center py-8 text-sm" style={{ color: 'var(--text-secondary)' }}>
            <p>© 2026 VeritasAI by The Verifiers. Built to fight digital deception.</p>
          </footer>
        </ThemeProvider>
      </body>
    </html>
  );
}
