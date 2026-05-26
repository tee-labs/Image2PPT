import { STATUS_LABEL } from "../utils/format";

export default function StatusPill({ status }: { status: string }) {
  return (
    <span className={`status ${status}`}>
      <span className="sdot" />
      {STATUS_LABEL[status] || status}
    </span>
  );
}
