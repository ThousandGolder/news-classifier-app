// Mobile-friendly enhancements
document.addEventListener("DOMContentLoaded", function () {
  // Example buttons functionality
  const exampleButtons = document.querySelectorAll(".example-btn");
  const textarea = document.getElementById("article_text");

  exampleButtons.forEach((button) => {
    button.addEventListener("click", function () {
      textarea.value = this.getAttribute("data-text");
      textarea.focus();

      // Show confirmation for mobile
      if (window.innerWidth < 768) {
        showMobileToast("Example text loaded!");
      }
    });
  });

  // Character count with mobile optimization
  if (textarea) {
    textarea.addEventListener("input", function () {
      const charCount = this.value.length;
      updateCharCount(charCount);

      // Auto-resize textarea on mobile
      if (window.innerWidth < 768) {
        this.style.height = "auto";
        this.style.height = this.scrollHeight + "px";
      }
    });

    // Initialize character count
    updateCharCount(textarea.value.length);
  }

  // Form submission loading state
  const form = document.querySelector("form");
  if (form) {
    form.addEventListener("submit", function () {
      const submitBtn = this.querySelector('button[type="submit"]');
      if (submitBtn && window.innerWidth < 768) {
        submitBtn.innerHTML = '<span class="loading"></span> Classifying...';
        submitBtn.disabled = true;
      }
    });
  }

  // Touch device enhancements
  if ("ontouchstart" in window) {
    document.body.classList.add("touch-device");

    // Improve button touch targets
    const buttons = document.querySelectorAll(".btn, .nav-link");
    buttons.forEach((btn) => {
      btn.style.minHeight = "44px";
      btn.style.minWidth = "44px";
    });
  }

  // Mobile menu close on click
  const navLinks = document.querySelectorAll(".nav-link");
  const navbarCollapse = document.querySelector(".navbar-collapse");

  navLinks.forEach((link) => {
    link.addEventListener("click", () => {
      if (window.innerWidth < 992) {
        const bsCollapse = new bootstrap.Collapse(navbarCollapse);
        bsCollapse.hide();
      }
    });
  });
});

function updateCharCount(count) {
  let charCountElement = document.getElementById("char-count");

  if (!charCountElement) {
    charCountElement = document.createElement("div");
    charCountElement.id = "char-count";
    charCountElement.className = "form-text";
    document
      .getElementById("article_text")
      .parentNode.appendChild(charCountElement);
  }

  charCountElement.textContent = `${count} characters`;

  if (count < 50) {
    charCountElement.style.color = "#dc3545";
    charCountElement.innerHTML = `${count} characters - <small>Need more text</small>`;
  } else if (count < 100) {
    charCountElement.style.color = "#fd7e14";
    charCountElement.innerHTML = `${count} characters - <small>Good, but more is better</small>`;
  } else {
    charCountElement.style.color = "#198754";
    charCountElement.innerHTML = `${count} characters - <small>Great! Ready to classify</small>`;
  }
}

function showMobileToast(message) {
  // Create a simple mobile-friendly toast notification
  const toast = document.createElement("div");
  toast.className = "position-fixed bottom-0 start-50 translate-middle-x mb-3";
  toast.innerHTML = `
        <div class="toast show" role="alert">
            <div class="toast-body bg-success text-white rounded">
                ${message}
            </div>
        </div>
    `;

  document.body.appendChild(toast);

  // Remove after 2 seconds
  setTimeout(() => {
    toast.remove();
  }, 2000);
}

// Handle orientation changes
window.addEventListener("orientationchange", function () {
  // Refresh character count on orientation change
  const textarea = document.getElementById("article_text");
  if (textarea) {
    updateCharCount(textarea.value.length);
  }
});

// Prevent zoom on input focus for iOS
document.addEventListener("touchstart", function () {}, { passive: true });
