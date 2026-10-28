// Vitest/jsdom bootstrap — DOM APIs the components touch but jsdom lacks.
if (typeof globalThis.ResizeObserver === 'undefined') {
  globalThis.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  } as unknown as typeof ResizeObserver;
}

// TestBed default providers (none needed yet — specs provide their own).
export default [];
