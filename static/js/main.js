document.addEventListener('DOMContentLoaded', () => {
  // Live preview in the Demo section with enhanced animation
  const fileInput = document.getElementById('imageInput');
  const previewWrap = document.getElementById('preview');
  const previewImg = document.getElementById('previewImage');
  const emptyState = document.getElementById('empty-state');
  const previewState = document.getElementById('preview-state');
  const previewDisplay = document.getElementById('preview-display');
  const uploadSpacer = document.getElementById('upload-spacer');
  
  if (fileInput && previewWrap && previewImg) {
    fileInput.addEventListener('change', (e) => {
      const file = e.target.files?.[0];
      if (!file) {
        previewWrap.classList.add('d-none');
        previewImg.removeAttribute('src');
        // Show empty state and upload spacer, hide preview state
        if (emptyState && previewState) {
          emptyState.classList.remove('d-none');
          previewState.classList.add('d-none');
        }
        if (uploadSpacer) {
          uploadSpacer.classList.remove('d-none');
        }
        return;
      }
      
      // Validate file type
      if (!file.type.startsWith('image/')) {
        alert('Please select a valid image file (PNG, JPG, JPEG)');
        fileInput.value = '';
        return;
      }
      
      // Validate file size (max 10MB)
      if (file.size > 10 * 1024 * 1024) {
        alert('File size too large. Please select an image under 10MB.');
        fileInput.value = '';
        return;
      }
      
      const reader = new FileReader();
      reader.onload = (ev) => {
        // Set left preview image
        previewImg.src = ev.target.result;
        previewWrap.classList.remove('d-none');
        
        // Hide upload spacer when image is selected
        if (uploadSpacer) {
          uploadSpacer.classList.add('d-none');
        }
        
        // Set right preview display
        if (previewDisplay) {
          previewDisplay.src = ev.target.result;
        }
        
        // Switch right panel from empty state to preview state
        if (emptyState && previewState) {
          emptyState.classList.add('d-none');
          previewState.classList.remove('d-none');
        }
        
        // Add smooth fade-in animation
        previewWrap.style.opacity = '0';
        if (previewState) previewState.style.opacity = '0';
        
        setTimeout(() => {
          previewWrap.style.transition = 'opacity 0.5s ease-in-out';
          previewWrap.style.opacity = '1';
          
          if (previewState) {
            previewState.style.transition = 'opacity 0.5s ease-in-out';
            previewState.style.opacity = '1';
          }
        }, 10);
      };
      reader.readAsDataURL(file);
    });
  }

  // Form validation with better UX
  const uploadForm = document.getElementById('upload-form');
  if (uploadForm) {
    uploadForm.addEventListener('submit', (e) => {
      if (!uploadForm.checkValidity()) {
        e.preventDefault();
        e.stopPropagation();
      }
      uploadForm.classList.add('was-validated');
      
      // Show loading state on submit button
      const submitBtn = uploadForm.querySelector('button[type="submit"]');
      if (submitBtn && uploadForm.checkValidity()) {
        const originalHTML = submitBtn.innerHTML;
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<i class="bi bi-hourglass-split me-2"></i>Processing...';
        
        // Reset after 30 seconds in case of error
        setTimeout(() => {
          submitBtn.disabled = false;
          submitBtn.innerHTML = originalHTML;
        }, 30000);
      }
    });
  }

  // Smooth scroll for anchor links
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      const href = this.getAttribute('href');
      if (href !== '#' && document.querySelector(href)) {
        e.preventDefault();
        const target = document.querySelector(href);
        const offsetTop = target.offsetTop - 80; // Account for fixed navbar
        window.scrollTo({
          top: offsetTop,
          behavior: 'smooth'
        });
      }
    });
  });

  // Add hover effect to cards
  const cards = document.querySelectorAll('.card');
  cards.forEach(card => {
    card.addEventListener('mouseenter', function() {
      this.style.transition = 'transform 0.3s ease, box-shadow 0.3s ease';
    });
  });

  // Navbar background change on scroll
  const navbar = document.querySelector('.navbar');
  if (navbar) {
    window.addEventListener('scroll', () => {
      if (window.scrollY > 50) {
        navbar.style.boxShadow = '0 6px 24px rgba(102, 126, 234, 0.4)';
      } else {
        navbar.style.boxShadow = '0 4px 20px rgba(102, 126, 234, 0.3)';
      }
    });
  }

  // Initialize tooltips if Bootstrap tooltips are used
  const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
  if (typeof bootstrap !== 'undefined') {
    tooltipTriggerList.map(function (tooltipTriggerEl) {
      return new bootstrap.Tooltip(tooltipTriggerEl);
    });
  }

  // Add animation to metrics cards on scroll into view
  const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -100px 0px'
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.style.opacity = '0';
        entry.target.style.transform = 'translateY(20px)';
        setTimeout(() => {
          entry.target.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
          entry.target.style.opacity = '1';
          entry.target.style.transform = 'translateY(0)';
        }, 100);
        observer.unobserve(entry.target);
      }
    });
  }, observerOptions);

  document.querySelectorAll('.metric-card').forEach(card => {
    observer.observe(card);
  });
});

