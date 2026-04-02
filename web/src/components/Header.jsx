import { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useTranslation } from 'react-i18next';

const NAV_KEYS = [
  {
    to: '/map',
    tKey: 'nav.realtimeMap',
    icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    ),
  },
  {
    to: '/how-it-works',
    tKey: 'nav.howItWorks',
    icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
      </svg>
    ),
  },
  {
    to: '/alerts',
    tKey: 'nav.alerts',
    icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
      </svg>
    ),
  },
  {
    to: '/history',
    tKey: 'nav.history',
    icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
      </svg>
    ),
  },
  {
    to: '/code',
    tKey: 'nav.code',
    icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
      </svg>
    ),
  },
];

// SVG flags — emoji flags don't render on many Linux/Windows systems
const FlagEN = () => (
  <svg className="w-5 h-3.5 rounded-sm overflow-hidden" viewBox="0 0 60 30">
    <clipPath id="s"><path d="M0,0 v30 h60 v-30 z"/></clipPath>
    <clipPath id="t"><path d="M30,15 h30 v15 z v15 h-30 z h-30 v-15 z v-15 h30 z"/></clipPath>
    <g clipPath="url(#s)"><path d="M0,0 v30 h60 v-30 z" fill="#012169"/><path d="M0,0 L60,30 M60,0 L0,30" stroke="#fff" strokeWidth="6"/><path d="M0,0 L60,30 M60,0 L0,30" clipPath="url(#t)" stroke="#C8102E" strokeWidth="4"/><path d="M30,0 v30 M0,15 h60" stroke="#fff" strokeWidth="10"/><path d="M30,0 v30 M0,15 h60" stroke="#C8102E" strokeWidth="6"/></g>
  </svg>
);
const FlagTR = () => (
  <svg className="w-5 h-3.5 rounded-sm" viewBox="0 0 60 40">
    <rect width="60" height="40" fill="#E30A17"/>
    <circle cx="22" cy="20" r="10" fill="#fff"/>
    <circle cx="25" cy="20" r="8" fill="#E30A17"/>
    <polygon points="33,20 28.5,16.5 30.5,21.5 26,18 32,18 27.5,21.5" fill="#fff"/>
  </svg>
);
const FlagJA = () => (
  <svg className="w-5 h-3.5 rounded-sm" viewBox="0 0 60 40">
    <rect width="60" height="40" fill="#fff"/>
    <circle cx="30" cy="20" r="10" fill="#BC002D"/>
  </svg>
);

const LANGS = [
  { code: 'en', Flag: FlagEN, label: 'EN' },
  { code: 'tr', Flag: FlagTR, label: 'TR' },
  { code: 'ja', Flag: FlagJA, label: 'JA' },
];

export default function Header() {
  const location = useLocation();
  const { t, i18n } = useTranslation();
  const [menuOpen, setMenuOpen] = useState(false);
  const [langOpen, setLangOpen] = useState(false);

  const currentLang = LANGS.find(l => i18n.language?.startsWith(l.code)) || LANGS[0];
  const CurrentFlag = currentLang.Flag;

  const switchLang = (code) => {
    i18n.changeLanguage(code);
    setLangOpen(false);
  };

  return (
    <header className="bg-zinc-900 border-b border-zinc-800 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex items-center justify-between h-14">

          {/* Logo */}
          <Link to="/" className="flex flex-col leading-tight shrink-0" onClick={() => setMenuOpen(false)}>
            <span className="text-base font-bold text-orange-500">{t('header.title')}</span>
            <span className="text-zinc-500 text-xs hidden sm:block">{t('header.subtitle')}</span>
          </Link>

          {/* Right: nav + lang + hamburger */}
          <div className="flex items-center gap-1">
            <nav className="hidden md:flex items-center gap-1">
              {NAV_KEYS.map(({ to, tKey, icon }) => {
                const active = location.pathname === to;
                return (
                  <Link
                    key={to}
                    to={to}
                    className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                      active
                        ? 'bg-orange-500/15 text-orange-400'
                        : 'text-zinc-400 hover:text-white hover:bg-zinc-800'
                    }`}
                  >
                    {icon}
                    {t(tKey)}
                  </Link>
                );
              })}
            </nav>

            {/* Language switcher */}
            <div className="relative">
              <button
                onClick={() => setLangOpen(v => !v)}
                className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium text-zinc-400 hover:text-white hover:bg-zinc-800 border border-zinc-700 hover:border-zinc-600 transition-colors"
                aria-label="Change language"
              >
                <CurrentFlag />
                <span>{currentLang.label}</span>
                <svg className={`w-3 h-3 transition-transform ${langOpen ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>
              {langOpen && (
                <>
                  <div className="fixed inset-0 z-40" onClick={() => setLangOpen(false)} />
                  <div className="absolute right-0 top-full mt-1 bg-zinc-800 border border-zinc-700 rounded-lg shadow-xl z-50 py-1 min-w-[120px]">
                    {LANGS.map(lang => (
                      <button
                        key={lang.code}
                        onClick={() => switchLang(lang.code)}
                        className={`w-full flex items-center gap-2.5 px-3 py-2.5 text-sm transition-colors ${
                          currentLang.code === lang.code
                            ? 'text-orange-400 bg-orange-500/10'
                            : 'text-zinc-300 hover:bg-zinc-700'
                        }`}
                      >
                        <lang.Flag />
                        <span>{t(`language.${lang.code}`)}</span>
                      </button>
                    ))}
                  </div>
                </>
              )}
            </div>

            {/* Hamburger — mobile only */}
            <button
              className="md:hidden p-2 rounded-lg text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors"
              onClick={() => setMenuOpen((v) => !v)}
              aria-label="Toggle menu"
            >
              {menuOpen ? (
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              ) : (
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              )}
            </button>
          </div>
        </div>

        {/* Mobile dropdown */}
        {menuOpen && (
          <nav className="md:hidden border-t border-zinc-800 py-2 flex flex-col gap-1">
            {NAV_KEYS.map(({ to, tKey, icon }) => {
              const active = location.pathname === to;
              return (
                <Link
                  key={to}
                  to={to}
                  onClick={() => setMenuOpen(false)}
                  className={`flex items-center gap-2 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                    active
                      ? 'bg-orange-500/15 text-orange-400'
                      : 'text-zinc-400 hover:text-white hover:bg-zinc-800'
                  }`}
                >
                  {icon}
                  {t(tKey)}
                </Link>
              );
            })}
          </nav>
        )}
      </div>
    </header>
  );
}
