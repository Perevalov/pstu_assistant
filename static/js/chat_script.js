(function () {
    var Message;
    Message = function (arg) {
        this.text = arg.text, this.message_side = arg.message_side;
        this.draw = function (_this) {
            return function () {
                var $message;
                $message = $($('.message_template').clone().html());
                $message.addClass(_this.message_side).find('.text').html(_this.text);
                $('.messages').append($message);
                return setTimeout(function () {
                    return $message.addClass('appeared');
                }, 0);
            };
        }(this);
        return this;
    };
    $(function () {
        var getMessageText, sendMessage;
        getMessageText = function () {
            var $message_input;
            $message_input = $('.message_input');
            return $message_input.val();
        };
        sendMessage = function (text,message_side) {
            var $messages, message;
            if (text.trim() === '') {
                return;
            }
            $('.message_input').val('');
            $messages = $('.messages');
            message = new Message({
                text: text,
                message_side: message_side
            });
            message.draw();
            return $messages.animate({ scrollTop: $messages.prop('scrollHeight') }, 300);
        };
        $('.send_message').click(function (e) {
            var rawText = getMessageText();
            sendMessage(rawText,'right');

            $.get("/get", { msg: rawText }).done(function(data) {
                sendMessage(data,'left');            
            });
            
        });

        $('.message_input').keyup(function (e) {
            if (e.which === 13) {
                var rawText = getMessageText();
                sendMessage(rawText,'right');

	        $.get("/get", { msg: rawText }).done(function(data) {
	            sendMessage(data,'left');            
         	});
            }
        });
        sendMessage('Привет! :)','left');
    });
}.call(this));
