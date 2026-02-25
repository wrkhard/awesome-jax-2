$(document).ready(function() {

  // Check if data exists
  if (typeof awesomeJaxData === 'undefined') {
    console.error('Data not found. Please run: npm run build');
    $('#softwareTable tbody').html(`
      <tr>
        <td colspan="6" class="text-center">
          <div class="alert alert-warning">
            No data found. Please run <code>npm run build</code> to generate data.
          </div>
        </td>
      </tr>
    `);
    return;
  }

  let table;
  let currentCategoryFilter = null;
  let currentStatusFilter = null;

  // Format date
  function formatDate(dateString) {
    if (!dateString) return 'Unknown';
    const date = new Date(dateString);
    const now = new Date();
    const diffTime = Math.abs(now - date);
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));

    if (diffDays < 1) {
      return 'Today';
    } else if (diffDays < 7) {
      return `${diffDays} days ago`;
    } else if (diffDays < 30) {
      const weeks = Math.floor(diffDays / 7);
      return `${weeks} week${weeks > 1 ? 's' : ''} ago`;
    } else if (diffDays < 365) {
      const months = Math.floor(diffDays / 30);
      return `${months} month${months > 1 ? 's' : ''} ago`;
    } else {
      const years = Math.floor(diffDays / 365);
      return `${years} year${years > 1 ? 's' : ''} ago`;
    }
  }

  // Format status badge
  function formatStatus(status) {
    const statusClass = `status-${status.toLowerCase()}`;
    const displayStatus = status === 'up-and-coming' ? 'Up & Coming' :
                         status.charAt(0).toUpperCase() + status.slice(1);
    return `<span class="status-badge ${statusClass}">${displayStatus}</span>`;
  }

  // Format stars with commas
  function formatStars(stars) {
    if (!stars) return '0';
    return stars.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  }

  // Populate table
  function populateTable() {
    const tbody = $('#softwareTable tbody');
    tbody.empty();

    awesomeJaxData.forEach(lib => {
      const rowClass = lib.status === 'inactive' ? 'inactive-row' : '';
      const githubUrl = lib.url;

      const row = $('<tr>').addClass(rowClass);

      // Library Name (with link)
      row.append($('<td>').html(
        `<a href="${githubUrl}" target="_blank" class="text-info">${lib.name}</a>`
      ));

      // Description
      row.append($('<td>').html(
        `<span class="description-cell">${lib.description || 'No description'}</span>`
      ));

      // Category
      row.append($('<td>').html(
        `<span class="category-badge">${lib.category}</span>`
      ));

      // Stars (with number or dash for unknown)
      const starsValue = lib.stars || 0;
      const starsDisplay = lib.stars
        ? `<span class="text-warning">★ ${formatStars(lib.stars)}</span>`
        : `<span class="text-muted">★ —</span>`;
      row.append($('<td>').attr('data-order', starsValue).html(starsDisplay));

      // Last Updated
      row.append($('<td>').html(
        `<span class="last-updated">${formatDate(lib.lastCommit)}</span>`
      ));

      // Status
      row.append($('<td>').html(formatStatus(lib.status)));

      tbody.append(row);
    });
  }

  // Initialize DataTable
  function initDataTable() {
    table = $('#softwareTable').DataTable({
      pageLength: 25,
      lengthMenu: [[10, 25, 50, 100, -1], [10, 25, 50, 100, "All"]],
      order: [[3, 'desc']], // Sort by stars by default
      columnDefs: [
        { orderable: true, targets: [0, 3, 4] },
        { orderable: false, targets: [1, 2, 5] },
        { type: 'num', targets: [3] }
      ],
      language: {
        search: "Search libraries:",
        lengthMenu: "Show _MENU_ libraries",
        info: "Showing _START_ to _END_ of _TOTAL_ libraries",
        paginate: {
          first: "First",
          last: "Last",
          next: "Next",
          previous: "Previous"
        }
      }
    });
  }

  // Populate filter dropdowns
  function populateFilters() {
    // Category filter
    const categories = [...new Set(awesomeJaxData.map(lib => lib.category))].sort();
    const categoryDropdown = $('#categoryDropdown');
    categoryDropdown.empty();
    categories.forEach(category => {
      categoryDropdown.append(
        $('<li>').append(
          $('<a>')
            .addClass('dropdown-item')
            .attr('href', '#')
            .text(category)
            .on('click', function(e) {
              e.preventDefault();
              filterByCategory(category);
            })
        )
      );
    });

    // Status filter - always show all options with counts
    const statusDropdown = $('#statusDropdown');
    statusDropdown.empty();

    const statusOrder = ['active', 'up-and-coming', 'inactive'];
    statusOrder.forEach(status => {
      const count = awesomeJaxData.filter(lib => lib.status === status).length;
      const displayStatus = status === 'up-and-coming' ? 'Up & Coming' :
                           status.charAt(0).toUpperCase() + status.slice(1);
      const item = $('<a>')
        .addClass('dropdown-item')
        .attr('href', '#')
        .text(`${displayStatus} (${count})`);

      if (count > 0) {
        item.on('click', function(e) {
          e.preventDefault();
          filterByStatus(status);
        });
      } else {
        item.addClass('disabled text-muted');
      }

      statusDropdown.append($('<li>').append(item));
    });
  }

  // Filter functions
  function filterByCategory(category) {
    currentCategoryFilter = category;
    applyFilters();
    $('#clearCategoryFilter').show();
  }

  function filterByStatus(status) {
    currentStatusFilter = status;
    applyFilters();
    $('#clearStatusFilter').show();
  }

  function applyFilters() {
    // Clear search
    table.search('');

    // Build filter function
    $.fn.dataTable.ext.search.pop(); // Remove any existing custom filter

    $.fn.dataTable.ext.search.push(function(settings, data, dataIndex) {
      const lib = awesomeJaxData[dataIndex];

      // Category filter
      if (currentCategoryFilter && lib.category !== currentCategoryFilter) {
        return false;
      }

      // Status filter
      if (currentStatusFilter && lib.status !== currentStatusFilter) {
        return false;
      }

      return true;
    });

    table.draw();
  }

  // Clear filter functions
  $('#clearCategoryFilter').on('click', function() {
    currentCategoryFilter = null;
    applyFilters();
    $(this).hide();
  });

  $('#clearStatusFilter').on('click', function() {
    currentStatusFilter = null;
    applyFilters();
    $(this).hide();
  });

  // Refresh data button
  $('#refreshData').on('click', function() {
    const btn = $(this);
    btn.prop('disabled', true).text('Building...');

    // Show instructions
    alert('To refresh data:\n\n1. Open terminal in the docs folder\n2. Run: npm run build\n3. Refresh this page\n\nFor faster builds without GitHub data:\nnpm run build:fast');

    btn.prop('disabled', false).text('Refresh Data');
  });

  // Initialize everything
  populateTable();
  initDataTable();
  populateFilters();

  // Hide clear buttons initially
  $('#clearCategoryFilter').hide();
  $('#clearStatusFilter').hide();

  // Display data statistics
  const stats = {
    total: awesomeJaxData.length,
    active: awesomeJaxData.filter(l => l.status === 'active').length,
    inactive: awesomeJaxData.filter(l => l.status === 'inactive').length,
    upAndComing: awesomeJaxData.filter(l => l.status === 'up-and-coming').length
  };

  console.log('Awesome JAX Statistics:', stats);
});