window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  },
  startup: {
    ready: function() {
      MathJax.startup.defaultReady();
      // Re-render MathJax when navigating with instant loading.
      // document$ is provided by the documentation theme's instant loading feature.
      if (typeof document$ !== "undefined") {
        document$.subscribe(function() {
          MathJax.typesetPromise();
        });
      }
    }
  }
};
