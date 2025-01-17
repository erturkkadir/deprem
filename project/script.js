$(document).ready(function() {
    $('#prediction-form').submit(function(event) {
        event.preventDefault();
        const location = $('#location').val();
        if (location.trim()) {
            // Simulate prediction response
            alert('Predicting for ' + location);
        } else {
            alert('Please enter a valid location.');
        }
    });

    $('#contact-form').submit(function(event) {
        event.preventDefault();
        const name = $('#name').val();
        const email = $('#email').val();
        const message = $('#message').val();
        if (name && email && message) {
            // Simulate sending form
            alert('Form submitted successfully!');
            $(this).trigger('reset');
        } else {
            alert('Please fill out all fields.');
        }
    });
});