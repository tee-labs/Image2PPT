export default function PageHead({
  title,
  crumb,
  actions,
}: {
  title: string;
  crumb?: string;
  actions?: React.ReactNode;
}) {
  return (
    <div className="page-head">
      <div>
        <div className="title">{title}</div>
        {crumb && <div className="crumb">{crumb}</div>}
      </div>
      <div className="spacer" />
      {actions}
    </div>
  );
}
