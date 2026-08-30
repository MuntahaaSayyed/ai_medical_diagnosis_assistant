/* =====================================================================
   main.js — global UI interactions for AI Medical Diagnosis Assistant
   ===================================================================== */

document.addEventListener("DOMContentLoaded", function () {
    // ---- Mobile sidebar toggle ----
    const toggleBtn = document.getElementById("sidebarToggle");
    const sidebar = document.querySelector(".sidebar");

    if (toggleBtn && sidebar) {
        toggleBtn.addEventListener("click", function () {
            sidebar.classList.toggle("sidebar-open");
        });

        // Close sidebar when clicking outside of it (mobile only)
        document.addEventListener("click", function (event) {
            const isClickInsideSidebar = sidebar.contains(event.target);
            const isClickOnToggle = toggleBtn.contains(event.target);
            if (!isClickInsideSidebar && !isClickOnToggle && window.innerWidth <= 768) {
                sidebar.classList.remove("sidebar-open");
            }
        });
    }

    // ---- Auto-dismiss flash messages after 5 seconds ----
    const flashMessages = document.querySelectorAll(".flash-msg");
    flashMessages.forEach(function (msg) {
        setTimeout(function () {
            msg.style.transition = "opacity 0.4s ease, transform 0.4s ease";
            msg.style.opacity = "0";
            msg.style.transform = "translateY(-8px)";
            setTimeout(function () {
                msg.remove();
            }, 400);
        }, 5000);
    });
});
