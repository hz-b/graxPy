document.addEventListener("DOMContentLoaded", () => {
  if (window.location.pathname.endsWith("/index.html") || window.location.pathname === "/") {
    document.body.classList.add("graxpy-home-page");
  }

  const footer = document.querySelector(".page footer");
  if (!footer) return;

  const notice = document.createElement("div");
  notice.className = "graxpy-site-notice";
  notice.innerHTML = `
    <p>GraxPy is developed by Helmholtz-Zentrum Berlin (HZB). Main developer and contact:
      <a href="mailto:simone.vadilonga@helmholtz-berlin.de">Simone Vadilonga</a>
      (<a href="mailto:simone.vadilonga@helmholtz-berlin.de">simone.vadilonga@helmholtz-berlin.de</a>).
    </p>
    <p>GraxPy is not yet at version 1.0.0. The API may therefore include breaking changes, which we try to avoid unless necessary.</p>`;
  footer.append(notice);
});
