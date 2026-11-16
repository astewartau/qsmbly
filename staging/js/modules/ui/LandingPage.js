/**
 * Landing Page Controller
 *
 * Manages the full-viewport welcome overlay (#landingPage) shown on top of the
 * app. The splash is always the front door: it shows on every plain visit. The
 * app itself lives at the "#app" hash (e.g. /qsmbly/#app), so launching just
 * navigates there and returning to the splash clears the hash.
 *
 * The overlay is hidden by toggling `landing-seen` on <html> (CSS handles the
 * actual display), which matches the FOUC-avoiding head script in index.html.
 */

const APP_HASH = '#app';

export class LandingPage {
  /**
   * @param {Object} opts
   * @param {() => void} [opts.onLaunch]      Called when launching normally.
   * @param {() => void} [opts.onLaunchTour]  Called when launching with the tour.
   */
  constructor({ onLaunch, onLaunchTour } = {}) {
    this.onLaunch = onLaunch;
    this.onLaunchTour = onLaunchTour;
    this.overlay = document.getElementById('landingPage');

    this._wire();
  }

  /** Show the landing overlay (e.g. when the logo is clicked). */
  show() {
    document.documentElement.classList.remove('landing-seen');
    if (location.hash === APP_HASH) {
      // drop "#app" from the URL without adding a history entry
      history.replaceState(null, '', location.pathname + location.search);
    }
    if (this.overlay) this.overlay.scrollTop = 0;
  }

  /** Hide the overlay and reveal the app (at the #app route). */
  hide() {
    document.documentElement.classList.add('landing-seen');
    if (location.hash !== APP_HASH) location.hash = APP_HASH;
  }

  _wire() {
    if (!this.overlay) return;

    const launchBtn = document.getElementById('landingLaunch');
    const tourBtn = document.getElementById('landingLaunchTour');
    const navLaunch = document.getElementById('landingNavLaunch');

    const launch = () => {
      this.hide();
      this.onLaunch?.();
    };

    launchBtn?.addEventListener('click', launch);
    navLaunch?.addEventListener('click', (e) => {
      e.preventDefault();
      launch();
    });

    tourBtn?.addEventListener('click', () => {
      this.hide();
      this.onLaunchTour?.();
    });
  }
}
