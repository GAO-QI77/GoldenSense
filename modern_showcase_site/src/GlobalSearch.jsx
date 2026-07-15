import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { CornerDownLeft, Search, X } from 'lucide-react';

import { querySearch, subscribeSearch } from './searchIndex';

const SOURCE_LABELS = {
  pages: '页面',
  quant: '量化面板',
  news: '近端新闻',
  prompts: '提问模板',
  indicators: '指标',
};

export default function GlobalSearch() {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [results, setResults] = useState(() => querySearch(''));
  const [activeIndex, setActiveIndex] = useState(0);
  const inputRef = useRef(null);
  const navigate = useNavigate();

  const refresh = useCallback(
    (nextQuery) => {
      setResults(querySearch(nextQuery));
      setActiveIndex(0);
    },
    [],
  );

  useEffect(() => subscribeSearch(() => refresh(query)), [refresh, query]);

  useEffect(() => {
    function onKeyDown(event) {
      const isShortcut = (event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k';
      if (isShortcut) {
        event.preventDefault();
        setOpen((prev) => !prev);
      }
      if (event.key === 'Escape') setOpen(false);
    }
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);

  useEffect(() => {
    if (open) {
      setQuery('');
      refresh('');
      const timer = setTimeout(() => inputRef.current?.focus(), 30);
      return () => clearTimeout(timer);
    }
    return undefined;
  }, [open, refresh]);

  function go(entry) {
    setOpen(false);
    if (!entry) return;
    if (entry.href) {
      window.open(entry.href, '_blank', 'noopener,noreferrer');
      return;
    }
    navigate(entry.route || '/');
    if (entry.hash) {
      setTimeout(() => {
        document.getElementById(entry.hash)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 120);
    }
  }

  function onInputKeyDown(event) {
    if (event.key === 'ArrowDown') {
      event.preventDefault();
      setActiveIndex((idx) => Math.min(idx + 1, results.length - 1));
    } else if (event.key === 'ArrowUp') {
      event.preventDefault();
      setActiveIndex((idx) => Math.max(idx - 1, 0));
    } else if (event.key === 'Enter') {
      event.preventDefault();
      go(results[activeIndex]);
    }
  }

  return (
    <>
      <button type="button" className="search-trigger" onClick={() => setOpen(true)}>
        <Search size={15} />
        <span>搜索研究内容</span>
        <kbd>⌘K</kbd>
      </button>

      {open && (
        <div className="search-overlay" role="dialog" aria-modal="true" aria-label="全局搜索">
          <div className="search-backdrop" onClick={() => setOpen(false)} />
          <div className="search-panel">
            <div className="search-input-row">
              <Search size={17} />
              <input
                ref={inputRef}
                value={query}
                placeholder="搜索面板、新闻、指标、提问模板…"
                onChange={(event) => {
                  setQuery(event.target.value);
                  refresh(event.target.value);
                }}
                onKeyDown={onInputKeyDown}
                aria-label="搜索"
              />
              <button type="button" className="search-close" onClick={() => setOpen(false)} aria-label="关闭搜索">
                <X size={16} />
              </button>
            </div>

            <ul className="search-results" role="listbox">
              {results.length === 0 && <li className="search-empty">没有匹配结果，换个关键词试试。</li>}
              {results.map((entry, index) => (
                <li key={entry.id || `${entry.source}-${index}`}>
                  <button
                    type="button"
                    className={index === activeIndex ? 'search-result active' : 'search-result'}
                    onMouseEnter={() => setActiveIndex(index)}
                    onClick={() => go(entry)}
                  >
                    <span className="search-result-source">{SOURCE_LABELS[entry.source] || entry.source}</span>
                    <span className="search-result-body">
                      <strong>{entry.title}</strong>
                      {entry.hint && <small>{entry.hint}</small>}
                    </span>
                    <CornerDownLeft size={14} className="search-result-go" />
                  </button>
                </li>
              ))}
            </ul>

            <footer className="search-footer">
              <span>↑↓ 选择 · Enter 跳转 · Esc 关闭</span>
              <span>索引覆盖：页面 / 量化面板 / 新闻 / 提问模板</span>
            </footer>
          </div>
        </div>
      )}
    </>
  );
}
