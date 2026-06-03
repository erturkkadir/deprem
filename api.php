<?php
  header('Access-Control-Allow-Origin: *');
  header('Access-Control-Allow-Methods: GET, POST, OPTIONS');
  header('Access-Control-Allow-Headers: Content-Type');

  if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
      exit;
  }

  $path = $_SERVER['PATH_INFO'] ?? '';
  $query = $_SERVER['QUERY_STRING'] ?? '';
  $api_url = 'https://building-upload-engaging-ongoing.trycloudflare.com/api' . $path;
  if ($query) {
      $api_url .= '?' . $query;
  }

  $ch = curl_init($api_url);
  curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
  curl_setopt($ch, CURLOPT_TIMEOUT, 30);
  curl_setopt($ch, CURLOPT_CONNECTTIMEOUT, 5);
  curl_setopt($ch, CURLOPT_CUSTOMREQUEST, $_SERVER['REQUEST_METHOD']);
  curl_setopt($ch, CURLOPT_FOLLOWLOCATION, true);

  if ($_SERVER['REQUEST_METHOD'] === 'POST') {
      curl_setopt($ch, CURLOPT_POSTFIELDS, file_get_contents('php://input'));
      curl_setopt($ch, CURLOPT_HTTPHEADER, ['Content-Type: application/json']);
  }

  $response = curl_exec($ch);
  $content_type = curl_getinfo($ch, CURLINFO_CONTENT_TYPE) ?: 'application/json';
  $http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
  curl_close($ch);

  http_response_code($http_code ?: 502);
  header('Content-Type: ' . $content_type);
  echo $response;
?>