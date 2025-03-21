import { ReactNode } from 'react';
import './Tooltip.css';

interface TooltipProps {
  text: string;
  children: ReactNode;
}

export const Tooltip = ({ text, children }: TooltipProps) => {
  return (
    <span className="info-icon">
      {children}
      <span className="tooltip-text">{text}</span>
    </span>
  );
};
