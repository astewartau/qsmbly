/**
 * Modal Manager Module
 *
 * Simple utility to manage modal open/close state and overlay-click-to-close behavior.
 * Focus handling (trap, Escape, return to opener) comes from dialogFocus.js.
 */

import { openDialog, closeDialog } from './dialogFocus.js';

export class ModalManager {
  /**
   * Create a modal manager for a specific modal element
   * @param {string} modalId - The DOM ID of the modal element
   */
  constructor(modalId) {
    this.modalId = modalId;
    this.modal = document.getElementById(modalId);

    if (this.modal) {
      this._setupOverlayClickClose();
    }
  }

  /**
   * Open the modal
   */
  open() {
    openDialog(this.modal, { onEscape: () => this.close() });
  }

  /**
   * Close the modal
   */
  close() {
    closeDialog(this.modal);
  }

  /**
   * Check if modal is currently open
   * @returns {boolean}
   */
  isOpen() {
    return this.modal?.classList.contains('active') ?? false;
  }

  /**
   * Toggle modal open/close state
   */
  toggle() {
    if (this.isOpen()) {
      this.close();
    } else {
      this.open();
    }
  }

  /**
   * Set up click-on-overlay-to-close behavior
   * @private
   */
  _setupOverlayClickClose() {
    this.modal.addEventListener('click', (e) => {
      // Only close if clicking the overlay itself, not modal content
      if (e.target === this.modal) {
        this.close();
      }
    });
  }
}
