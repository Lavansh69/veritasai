'use client';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { motion } from 'framer-motion';
import { useTheme } from './ThemeProvider';
import { Shield, Sun, Moon, Menu, X } from 'lucide-react';
import { useState } from 'react';

const links = [
  { href: '/', label: 'Home' },
  { href: '/upload', label: 'Upload' },
  { href: '/statistics', label: 'Statistics' },
];

export default function Navbar() {
  const pathname = usePathname();
  const { theme, toggle } = useTheme();
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <motion.nav
      initial={{ y: -80, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.6, ease: 'easeOut' }}
      className="fixed top-0 left-0 right-0 z-50"
    >
      <div className="glass-card mx-4 mt-3 px-6 py-3 flex items-center justify-between">
        {/* Logo */}
        <Link href="/" className="flex items-center gap-2 group">
          <motion.div whileHover={{ rotate: 360 }} transition={{ duration: 0.6 }}>
            <Shield className="w-7 h-7 text-brand-400" />
          </motion.div>
          <span className="font-display font-bold text-xl gradient-text">VeritasAI</span>
        </Link>

        {/* Desktop Links */}
        <div className="hidden md:flex items-center gap-8">
          {links.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className={`nav-link text-sm font-medium transition-colors duration-200 ${
                pathname === link.href
                  ? 'text-brand-400'
                  : 'hover:text-brand-300'
              }`}
              style={{ color: pathname === link.href ? 'var(--accent)' : 'var(--text-secondary)' }}
            >
              {link.label}
            </Link>
          ))}
        </div>

        {/* Actions */}
        <div className="flex items-center gap-3">
          <button
            onClick={toggle}
            className="p-2 rounded-lg transition-colors"
            style={{ color: 'var(--text-secondary)' }}
            aria-label="Toggle theme"
          >
            {theme === 'dark' ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
          </button>

          <Link href="/upload" className="hidden md:block btn-primary text-sm !px-4 !py-2">
            Analyze Media
          </Link>

          <button
            className="md:hidden p-2"
            onClick={() => setMobileOpen(!mobileOpen)}
            style={{ color: 'var(--text-secondary)' }}
            aria-label="Toggle menu"
          >
            {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>
      </div>

      {/* Mobile menu */}
      {mobileOpen && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-card mx-4 mt-2 p-4 md:hidden"
        >
          {links.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              onClick={() => setMobileOpen(false)}
              className="block py-2 text-sm font-medium"
              style={{ color: pathname === link.href ? 'var(--accent)' : 'var(--text-secondary)' }}
            >
              {link.label}
            </Link>
          ))}
        </motion.div>
      )}
    </motion.nav>
  );
}
