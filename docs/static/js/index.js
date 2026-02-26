// ===== Nerfies-style TempoRAL Project Page — JavaScript =====

$(document).ready(function() {

  // Navbar burger toggle (Nerfies pattern)
  $(".navbar-burger").click(function() {
    $(".navbar-burger").toggleClass("is-active");
    $(".navbar-menu").toggleClass("is-active");
  });

  // BibTeX copy button injection
  var bibtexSection = document.getElementById('BibTeX');
  if (bibtexSection) {
    var preBlock = bibtexSection.querySelector('pre');
    if (preBlock) {
      var copyBtn = document.createElement('button');
      copyBtn.className = 'bibtex-copy-btn';
      copyBtn.innerHTML = '<i class="fas fa-copy"></i> Copy';
      copyBtn.addEventListener('click', function() {
        var code = preBlock.querySelector('code');
        navigator.clipboard.writeText(code.textContent).then(function() {
          copyBtn.innerHTML = '<i class="fas fa-check"></i> Copied!';
          setTimeout(function() {
            copyBtn.innerHTML = '<i class="fas fa-copy"></i> Copy';
          }, 2000);
        });
      });
      preBlock.appendChild(copyBtn);
    }
  }

});
