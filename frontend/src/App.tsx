import { useEffect, useState } from 'react'
import ChatInterface from './components/ChatInterface'
import HomePage from './components/HomePage'
import './App.css'

type ThemeMode = 'dark' | 'light'

function getCanvasIdFromUrl() {
  try {
    const url = new URL(window.location.href)
    return url.searchParams.get('canvasId') || ''
  } catch {
    return ''
  }
}

function readInitialTheme(): ThemeMode {
  try {
    const stored = localStorage.getItem('polystudio:theme')
    if (stored === 'dark' || stored === 'light') return stored
  } catch {
    // ignore
  }
  return 'dark'
}

function App() {
  const [canvasId, setCanvasId] = useState<string>(() => getCanvasIdFromUrl())
  const [theme, setTheme] = useState<ThemeMode>(() => readInitialTheme())

  useEffect(() => {
    const onPop = () => setCanvasId(getCanvasIdFromUrl())
    window.addEventListener('popstate', onPop)
    return () => window.removeEventListener('popstate', onPop)
  }, [])

  useEffect(() => {
    try {
      document.documentElement.dataset.theme = theme
      localStorage.setItem('polystudio:theme', theme)
    } catch {
      // ignore
    }
  }, [theme])

  const toggleTheme = () => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))

  return (
    <div className="app">
      {canvasId ? (
        <ChatInterface
          initialCanvasId={canvasId}
          theme={theme}
          onToggleTheme={toggleTheme}
          onSetTheme={setTheme}
        />
      ) : (
        <HomePage theme={theme} onToggleTheme={toggleTheme} />
      )}
    </div>
  )
}

export default App







