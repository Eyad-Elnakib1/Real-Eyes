import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/**
 * Safely parses the FastAPI error response detail. 
 * If it's an array of Pydantic validation objects, it extracts the first message.
 * If it's a string, it returns it directly.
 * Otherwise, it falls back to the default message.
 */
export function formatApiError(defaultMessage: string, error?: any): string {
  if (!error || !error.response || !error.response.data) {
    return defaultMessage;
  }

  const detail = error.response.data.detail;

  if (typeof detail === 'string') {
    return detail;
  }

  // Handle Pydantic validation error array: [{type, loc, msg, input}]
  if (Array.isArray(detail) && detail.length > 0 && detail[0].msg) {
    const loc = detail[0].loc ? detail[0].loc[detail[0].loc.length - 1] : 'Field';
    return `${loc}: ${detail[0].msg}`;
  }

  return defaultMessage;
}
