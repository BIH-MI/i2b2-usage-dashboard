import { useTheme } from '../theme';
import { useUserFilter } from '../context/UserFilterContext';
import InfoTooltip from './InfoTooltip';

// page heading with a line that says whose data the charts show
function PageTitle({ title }) {
  const { colors } = useTheme();
  const { selectedUsers } = useUserFilter();

  const notice =
    selectedUsers.length === 0
      ? 'Showing all users. To see specific users, use Select Users in the menu.'
      : `Showing ${selectedUsers.length} selected user${selectedUsers.length > 1 ? 's' : ''}. Unselect all in Select Users to see everyone again.`;

  return (
    <div style={{ marginBottom: '1rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <h2 style={{ color: colors.textOnPage, margin: 0 }}>{title}</h2>
        <InfoTooltip
          corner={false}
          text="All charts on this page show every user by default. To look at specific users, open Select Users in the menu on the left. Exclude groups removes a whole group, also from a selection. The date range above the charts applies to all of them."
        />
      </div>
      <p style={{ color: `${colors.textOnPage}99`, fontSize: '0.85rem', margin: '0.3rem 0 0' }}>{notice}</p>
    </div>
  );
}

export default PageTitle;
