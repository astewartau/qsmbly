/**
 * A field kept as typed until it is committed follows its value when the value changes from
 * outside (undo, reset, another control), and only then: not on mount, which would overwrite
 * typing that started before the first effect ran.
 */
import { useEffect, useRef } from 'preact/hooks';

/**
 * @template T
 * @param {T} value
 * @param {(value: T) => void} follow
 */
export function useFollow(value, follow) {
  const last = useRef(value);
  useEffect(() => {
    if (last.current === value) return;
    last.current = value;
    follow(value);
  }, [value]);
}
