import { toDateParam } from './formatDate';

// userIds is optional, added as repeated ?user_id=... params
export function buildDateRangeParams(startDate, endDate, userIds = []) {
  const params = new URLSearchParams({
    start: toDateParam(startDate),
    end: toDateParam(endDate)
  });
  userIds.forEach((id) => params.append('user_id', id));
  return params;
}
