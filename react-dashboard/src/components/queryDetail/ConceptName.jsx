import { useTheme } from '../../theme';
import { splitConceptPath } from '../../utils/conceptPath';

// "folder › folder › name" 
function ConceptName({ concept }) {
  const { colors } = useTheme();
  const { prefix, label } = splitConceptPath(concept);
  return (
    <span title={concept} style={{ color: colors.textOnPage }}>
      {prefix && <span style={{ color: `${colors.textOnPage}99` }}>{prefix} › </span>}
      {label}
    </span>
  );
}

export default ConceptName;
