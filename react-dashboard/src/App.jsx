import { Routes, Route, Navigate } from 'react-router-dom';
import Sidebar from './navigation/Sidebar';
import UserMenu from './components/UserMenu';
import DemoBanner from './components/DemoBanner';
import RequireAuth from './components/RequireAuth';
import UsersPage from './pages/UsersPage';
import QueriesPage from './pages/QueriesPage';
import QueryDetailPage from './pages/QueryDetailPage';
import LoginPage from './pages/LoginPage';
import SettingsPage from './pages/SettingsPage';
import { UserFilterProvider } from './context/UserFilterContext';
import { DateRangeProvider } from './context/DateRangeContext';

function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      {/* the user filter and date range load data, so they sit inside the login check */}
      <Route
        path="/*"
        element={
          <RequireAuth>
            <UserFilterProvider>
              <DateRangeProvider>
                <div style={{ display: 'flex' }}>
                  <Sidebar />
                  {/* relative, so the account button sits in the top right corner of the page area */}
                  <div style={{ flex: 1, minWidth: 0, position: 'relative' }}>
                    <DemoBanner />
                    <UserMenu />
                    <Routes>
                      <Route path="/" element={<Navigate to="/users" replace />} />
                      <Route path="/users" element={<UsersPage />} />
                      <Route path="/queries" element={<QueriesPage />} />
                      <Route path="/queries/:queryId" element={<QueryDetailPage />} />
                      <Route path="/settings" element={<SettingsPage />} />
                    </Routes>
                  </div>
                </div>
              </DateRangeProvider>
            </UserFilterProvider>
          </RequireAuth>
        }
      />
    </Routes>
  );
}

export default App;
