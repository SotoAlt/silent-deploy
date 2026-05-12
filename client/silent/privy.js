// Privy auth integration for the federation panel.
//
// Lazy-loads React + Privy SDK via esm.sh on first use. If the CDN
// fails, the federation panel stays usable in anonymous mode (just no
// login UI). The hub-side PrivyIdentityProvider falls back to
// self-asserted when `privy_token` is absent in the WS hello.
//
// Exposes:
//   window.AuraPrivy.getAccessToken() → Promise<string | null>
//   window.AuraPrivy.isReady()        → boolean
//   window.AuraPrivy.isLoggedIn()     → boolean
//   window.AuraPrivy.user()           → user object | null
//
// Mounted into the element id="privy-mount" inside the federation panel.

const PRIVY_APP_ID = 'cmp2vbmpd000z0ckzcylyu17n';
const REACT_VERSION = '18.3.1';
const PRIVY_VERSION = '2';   // major-version pin

let _ready = false;
let _authenticated = false;
let _user = null;
let _getAccessTokenImpl = null;

function _emitState() {
  // Notify the federation panel (train.js) that auth state changed so
  // it can re-render the "<email> · X rounds" subtitle. Cheap event;
  // train.js subscribes with addEventListener('aura-privy-changed').
  window.dispatchEvent(new CustomEvent('aura-privy-changed', {
    detail: { ready: _ready, authenticated: _authenticated, user: _user },
  }));
}

window.AuraPrivy = {
  isReady: () => _ready,
  isLoggedIn: () => _authenticated,
  user: () => _user,
  async getAccessToken() {
    if (!_getAccessTokenImpl) return null;
    try {
      return await _getAccessTokenImpl();
    } catch (e) {
      console.warn('[privy] getAccessToken failed:', e);
      return null;
    }
  },
};

async function _bootstrap() {
  const mount = document.getElementById('privy-mount');
  if (!mount) {
    // Page doesn't have a Privy mount — federation panel is absent
    // or this script loaded somewhere unexpected. No-op.
    return;
  }

  let React, ReactDOM, PrivyAuth;
  try {
    [React, ReactDOM, PrivyAuth] = await Promise.all([
      import(`https://esm.sh/react@${REACT_VERSION}`),
      import(`https://esm.sh/react-dom@${REACT_VERSION}/client`),
      import(`https://esm.sh/@privy-io/react-auth@${PRIVY_VERSION}?deps=react@${REACT_VERSION},react-dom@${REACT_VERSION}`),
    ]);
  } catch (e) {
    // CDN load failed (offline, blocked, etc.). Render a small
    // notice so it's not silent, but don't block the federation
    // panel — anonymous mode still works.
    console.warn('[privy] SDK load failed; staying in anonymous mode:', e);
    mount.textContent = 'sign-in unavailable (offline)';
    mount.style.color = 'var(--dim, #6b7585)';
    mount.style.fontSize = '11px';
    return;
  }

  const h = React.createElement;
  const { PrivyProvider, usePrivy } = PrivyAuth;

  function AuthWidget() {
    const privy = usePrivy();
    React.useEffect(() => {
      _ready = privy.ready;
      _authenticated = privy.authenticated;
      _user = privy.user;
      _getAccessTokenImpl = privy.getAccessToken;
      _emitState();
    }, [privy.ready, privy.authenticated, privy.user]);

    if (!privy.ready) {
      return h('span', {style: {fontSize: '11px', color: 'var(--dim, #6b7585)'}},
                'loading sign-in…');
    }
    if (privy.authenticated) {
      const email = (privy.user && privy.user.email && privy.user.email.address)
                     || (privy.user && privy.user.wallet && privy.user.wallet.address)
                     || 'signed in';
      const masked = email.includes('@')
                      ? email.charAt(0) + '***@' + email.split('@')[1]
                      : email.slice(0, 6) + '…' + email.slice(-4);
      return h('span', {style: {fontSize: '11px', display: 'inline-flex',
                                  gap: '8px', alignItems: 'center'}},
        h('span', {style: {color: 'var(--cyan, #59c2ff)'}}, masked),
        h('button', {
          onClick: () => privy.logout(),
          style: {background: 'transparent',
                  border: '1px solid var(--border, #1d242e)',
                  color: 'var(--dim, #6b7585)',
                  padding: '2px 8px', borderRadius: '4px',
                  cursor: 'pointer', fontSize: '10px',
                  fontFamily: 'inherit'},
        }, 'logout'));
    }
    return h('button', {
      onClick: () => privy.login(),
      style: {background: 'transparent',
              border: '1px solid var(--cyan, #59c2ff)',
              color: 'var(--cyan, #59c2ff)',
              padding: '4px 10px', borderRadius: '4px',
              cursor: 'pointer', fontSize: '11px',
              fontFamily: 'inherit'},
    }, 'sign in to track');
  }

  const root = ReactDOM.createRoot(mount);
  root.render(
    h(PrivyProvider, {
      appId: PRIVY_APP_ID,
      config: {
        appearance: {theme: 'dark', accentColor: '#59c2ff'},
        // Match dashboard-side login methods (silent/relay/lepong share
        // one app). Without this, Privy's modal shows ALL methods enabled
        // server-side, which works but is noisy.
        loginMethods: ['email', 'google', 'wallet'],
      },
    }, h(AuthWidget)),
  );
}

// Defer to DOMContentLoaded so the mount div definitely exists.
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', _bootstrap);
} else {
  _bootstrap();
}
