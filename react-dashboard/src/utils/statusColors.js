// query status -> color, used by the status chart and the query detail page
export function statusColorMap(colors) {
  return {
    FINISHED: colors.status.good,
    COMPLETED: colors.status.good,
    CANCELLED: colors.status.warning,
    INCOMPLETE: colors.status.warning,
    TIMEDOUT: colors.status.serious,
    ERROR: colors.status.critical,
    QUEUED: colors.status.neutral,
    PROCESSING: colors.status.neutral,
    MEDIUM_QUEUE: colors.status.neutral,
    LARGE_QUEUE: colors.status.neutral
  };
}
