// src/components/Panel.jsx
import { C } from '../styles/colors';

export function Panel({ children, style = {} }) {
  return (
    <div style={{
      background: C.panel,
      border: `1px solid ${C.border}`,
      borderRadius: 8,
      padding: 14,
      ...style,
    }}>
      {children}
    </div>
  );
}