import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Search } from 'lucide-react';
import { useTheme } from '../theme';

function QuerySearch({ isCollapsed = false }) {
  const { colors } = useTheme();
  const navigate = useNavigate();
  const [queryId, setQueryId] = useState('');
  const [error, setError] = useState('');

  function goToQuery() {
    const trimmed = queryId.trim().replace(/\s+/g, ''); // "3 5" becomes "35"
    if (!trimmed) return;

    if (!/^\d+$/.test(trimmed)) {
      setError('ID should only contain numbers');
      return;
    }

    setError('');
    navigate(`/queries/${trimmed}`);
  }

  if (isCollapsed) {
    return null; 
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', padding: '0.5rem 0.75rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', color: colors.textOnPage, fontSize: '0.9rem' }}>
        <Search size={18} />
        <span>Select Query</span>
      </div>

      <div style={{ position: 'relative' }}>
        <input
          value={queryId}
          onChange={(e) => {
            setQueryId(e.target.value);
            setError('');
          }}
          onKeyDown={(e) => {
            if (e.key === 'Enter') goToQuery();
          }}
          placeholder="Query ID..."
          style={{
            width: '100%',
            boxSizing: 'border-box',
            border: `1.5px solid ${error ? colors.status.critical : `${colors.tileBorder}33`}`,
            borderRadius: '8px',
            outline: 'none',
            padding: '0.4rem 2rem 0.4rem 0.6rem',
            backgroundColor: 'transparent',
            color: colors.textOnPage,
            fontSize: '0.85rem'
          }}
        />
        <Search
          size={15}
          color={`${colors.textOnPage}99`}
          onClick={goToQuery}
          style={{ position: 'absolute', right: '0.5rem', top: '50%', transform: 'translateY(-50%)', cursor: 'pointer' }}
        />
      </div>

      {error && <span style={{ color: colors.status.critical, fontSize: '0.75rem' }}>{error}</span>}
    </div>
  );
}

export default QuerySearch;
